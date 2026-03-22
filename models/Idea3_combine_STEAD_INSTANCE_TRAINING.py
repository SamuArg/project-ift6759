"""
Debated with claude:
INSTANCE → STEAD generalization: is it a good idea?
It's a legitimate experiment but the asymmetry works against you. STEAD is a global dataset — it contains teleseismic events, regional events, a huge range of distances and magnitudes, and noise from many different tectonic and network environments. INSTANCE is a regional Italian network with a much more homogeneous noise profile and event distribution.
Training on INSTANCE and testing on STEAD means your model has to generalize from a narrow distribution to a wide one. That's hard. The reverse (STEAD → INSTANCE) is the standard benchmark in the literature for a reason — STEAD's diversity makes it a reasonable prior for most deployments. If you really want to test generalization, I'd suggest training on INSTANCE and evaluating on a geographically distinct regional dataset (TXED, GEOFON subset, PNW) rather than all of STEAD. That would be a cleaner and more publishable generalization claim.
That said — if your goal is ultimately an ensemble, this actually doesn't matter much. You want each sub-model to be strong, not necessarily generalizable in isolation.
"""


"""
STEP 3 — U-Net + BiLSTM + detection head + multi-dataset training
=================================================================
Builds directly on Step 2. The model architecture is IDENTICAL.
The entire change is in the data pipeline: we train jointly on
STEAD + INSTANCE instead of STEAD alone.
 
WHY multi-dataset training is the single biggest generalization gain:
    STEAD and INSTANCE have fundamentally different characteristics:
 
    STEAD:
        - ~1.2M traces, global coverage
        - Clean, professionally annotated picks
        - Dominated by tectonic earthquakes, M0-M8
        - Noise traces are mostly broadband noise
        - Pick precision: ~0.01-0.05s
 
    INSTANCE:
        - ~1.3M traces, Italian regional network (INGV)
        - More realistic operational noise (anthropogenic, cultural)
        - More local/regional events, different magnitude distribution
        - Noisier P-picks, especially for weak events
        - Contains induced seismicity
 
    A model trained only on STEAD learns:
        - STEAD's noise floor and spectral characteristics
        - STEAD's typical waveform morphology (teleseismic + regional global)
        - STEAD's pick style (high-precision analyst picks)
 
    When deployed on a regional Italian network (or validated on INSTANCE),
    it sees unfamiliar noise and different waveform shapes — and fails.
 
    Joint training forces the backbone to learn INVARIANT features:
        - The characteristic P-wave first motion shape
        - The amplitude step at phase onset
        - The spectral change between noise and signal
 
    Because these features must work for BOTH datasets, the model cannot
    overfit to either dataset's specific noise characteristics.
 
    This is arguably the most impactful change you can make at this stage.
    Architectural improvements help at the margins; dataset diversity helps
    fundamentally.
 
What changes vs Step 2:
    1. MultiDatasetWrapper concatenates STEAD + INSTANCE generators.
    2. Each dataset uses its own SeisBenchPipelineWrapper.
    3. The combined DataLoader samples from the merged generator.
    4. Everything else (model, loss, training loop) is identical to Step 2.
 
Key implementation decisions:
 
    SAMPLING STRATEGY — equal mixing vs proportional:
        We use weighted sampling to draw roughly equal numbers of traces
        from each dataset per batch (50/50), rather than proportional to
        dataset size. WHY: if we sampled proportionally, the larger dataset
        would dominate training. Equal mixing ensures the model sees enough
        INSTANCE traces to actually learn from them.
        This is implemented via WeightedRandomSampler.
 
    NORMALIZATION — already per-trace:
        Because we normalize each trace independently (std normalization
        in SeisBench), there is no amplitude distribution mismatch between
        datasets. A trace from STEAD and a trace from INSTANCE both have
        unit variance after normalization. No dataset-level normalization
        is needed.
 
    LABEL SIGMA — kept at 10 for both:
        INSTANCE picks are noisier (~0.1-0.2s uncertainty vs ~0.05s for STEAD).
        A sigma=10 (0.1s) Gaussian is a reasonable compromise.
        In a later iteration you could use sigma=20 for INSTANCE and sigma=10
        for STEAD, but the benefit is marginal compared to mixing the data.
 
    METADATA COLUMN NAMES — handled per dataset:
        STEAD uses "trace_p_arrival_sample"; INSTANCE may use the same or
        "trace_P_arrival_sample". The SeisBenchPipelineWrapper already handles
        this with its column detection logic.
"""
 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
 
import seisbench.data as sbd
import seisbench.generate as sbg
 
 
# ──────────────────────────────────────────────────────────────────────────────
# DATA PIPELINE  (Step 2 wrapper, reused per-dataset)
# ──────────────────────────────────────────────────────────────────────────────
 
class SeisBenchPipelineWrapper:
    """
    Single-dataset SeisBench wrapper (unchanged from Step 2).
    Used twice in Step 3: once for STEAD, once for INSTANCE.
    """
    def __init__(self, dataset_name="STEAD", split="train", model_type="eqtransformer",
                 component_order="ZNE", max_distance=None, transformation_shape="gaussian",
                 transformation_sigma=10, dataset_fraction=1.0):
        self.dataset_name         = dataset_name.upper()
        self.split                = split.lower()
        self.model_type           = model_type.lower()
        self.component_order      = component_order
        self.max_distance         = max_distance
        self.transformation_shape = transformation_shape
        self.transformation_sigma = transformation_sigma
        self.dataset_fraction     = dataset_fraction
 
        if self.dataset_name == "STEAD":
            dataset = sbd.STEAD(component_order=self.component_order)
        elif self.dataset_name == "INSTANCE":
            dataset = sbd.InstanceCountsCombined(component_order=self.component_order)
        elif self.dataset_name == "VCSEIS":
            dataset = sbd.VCSEIS(component_order=self.component_order)
        elif self.dataset_name == "GEOFON":
            dataset = sbd.GEOFON(component_order=self.component_order)
        elif self.dataset_name == "TXED":
            dataset = sbd.TXED(component_order=self.component_order)
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported.")
 
        if self.split == "train":
            self.dataset = dataset.train()
        elif self.split in ["dev", "val", "validation"]:
            self.dataset = dataset.dev()
        elif self.split == "test":
            self.dataset = dataset.test()
 
        if self.max_distance is not None and self.dataset_name in ["STEAD", "INSTANCE"]:
            possible_cols = ["path_hyp_distance_km", "source_distance_km"]
            dist_col = next((c for c in possible_cols if c in self.dataset.metadata.columns), None)
            if dist_col:
                mask = (
                    (self.dataset.metadata[dist_col] <= self.max_distance) |
                    (self.dataset.metadata[dist_col].isna())
                ).values
                self.dataset.filter(mask)
 
        if 0.0 < self.dataset_fraction < 1.0:
            total = len(self.dataset)
            n     = int(total * self.dataset_fraction)
            mask  = np.zeros(total, dtype=bool)
            mask[np.random.choice(total, n, replace=False)] = True
            self.dataset.filter(mask)
 
        self.generator = sbg.GenericGenerator(self.dataset)
        self._attach_pipeline()
 
    def _attach_pipeline(self):
        window_len = 6000 if self.model_type == "eqtransformer" else 3001
 
        p_col = "trace_p_arrival_sample"
        s_col = "trace_s_arrival_sample"
        if p_col not in self.dataset.metadata.columns:
            p_col = "trace_P_arrival_sample"
            s_col = "trace_S_arrival_sample"
 
        augmentations = [
            sbg.WindowAroundSample(
                metadata_keys=[p_col, s_col], selection="random",
                samples_before=window_len // 2, strategy="pad", windowlen=window_len,
            ),
            sbg.ChangeDtype(np.float32),
            sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="std"),
            sbg.ProbabilisticLabeller(
                label_columns=[p_col], shape=self.transformation_shape,
                sigma=self.transformation_sigma, key=("X", "y_p"), dim=0,
            ),
            sbg.ChangeDtype(np.float32, key="y_p"),
            sbg.ProbabilisticLabeller(
                label_columns=[s_col], shape=self.transformation_shape,
                sigma=self.transformation_sigma, key=("X", "y_s"), dim=0,
            ),
            sbg.ChangeDtype(np.float32, key="y_s"),
            sbg.ChangeDtype(np.float32, key="X"),
            sbg.DetectionLabeller(
                p_phases=[p_col], s_phases=[s_col], key=("X", "y_det"),
            ),
            sbg.ChangeDtype(np.float32, key="y_det"),
        ]
        self.generator.add_augmentations(augmentations)
 
    def get_dataloader(self, batch_size=32, num_workers=4, shuffle=True):
        return DataLoader(self.generator, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# MULTI-DATASET DATALOADER  (NEW in Step 3)
# ──────────────────────────────────────────────────────────────────────────────
 
def build_multi_dataset_loader(
    dataset_configs: list[dict],
    batch_size: int   = 32,
    num_workers: int  = 4,
    shuffle: bool     = True,
    equal_mixing: bool = True,
) -> DataLoader:
    """
    Combines multiple SeisBench datasets into a single DataLoader with
    controlled mixing ratios.
 
    Args:
        dataset_configs: list of dicts, each passed as kwargs to
                         SeisBenchPipelineWrapper. Must include at minimum
                         'dataset_name' and 'split'.
        equal_mixing:    if True, each dataset contributes equally to each
                         batch regardless of its absolute size.
                         if False, sampling is proportional to dataset size.
 
    WHY WeightedRandomSampler vs ConcatDataset + shuffle:
        PyTorch's ConcatDataset + shuffle gives proportional sampling —
        a 1.2M-trace STEAD will dominate a 400K-trace subset of INSTANCE.
        WeightedRandomSampler lets us assign per-sample weights so each
        dataset gets exactly the fraction of batches we want.
 
    WHY equal_mixing=True as default:
        At 50/50, the model sees comparable amounts of each dataset per
        epoch regardless of their size difference. This prevents the larger
        dataset from overwhelming the smaller one's gradient signal.
        If you have strong prior knowledge that one dataset is higher quality,
        you could tune this ratio (e.g., 70% STEAD, 30% INSTANCE).
    """
    wrappers = []
    for cfg in dataset_configs:
        wrapper = SeisBenchPipelineWrapper(**cfg)
        wrappers.append(wrapper)
 
    # ConcatDataset joins the generators end-to-end
    combined = ConcatDataset([w.generator for w in wrappers])
 
    if equal_mixing:
        # Each dataset gets weight = 1/num_datasets regardless of its size.
        # Per-sample weight = dataset_weight / dataset_size.
        n_datasets = len(wrappers)
        weights = []
        for w in wrappers:
            n = len(w.generator)
            # Each sample in this dataset gets weight = 1 / (n_datasets * n)
            weights.extend([1.0 / (n_datasets * n)] * n)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(combined),
            replacement=True,   # replacement=True required for WeightedRandomSampler
        )
        return DataLoader(
            combined,
            batch_size=batch_size,
            sampler=sampler,          # sampler replaces shuffle
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        # Proportional sampling — just use shuffle
        return DataLoader(
            combined,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
 
 
# ──────────────────────────────────────────────────────────────────────────────
# MODEL  (IDENTICAL to Step 2 — no architectural changes)
# ──────────────────────────────────────────────────────────────────────────────
 
class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(channels)
        self.bn2   = nn.BatchNorm1d(channels)
 
    def forward(self, x):
        r = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + r)
 
 
class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        self.res  = ResidualConvBlock(in_ch, kernel_size=kernel_size, dilation=dilation)
        self.down = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )
 
    def forward(self, x):
        x    = self.res(x)
        skip = x
        x    = self.down(x)
        return x, skip
 
 
class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, kernel_size: int = 7):
        super().__init__()
        self.merge = nn.Sequential(
            nn.Conv1d(in_ch + skip_ch, out_ch, kernel_size=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )
        self.res = ResidualConvBlock(out_ch, kernel_size=kernel_size, dilation=1)
 
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.merge(x)
        x = self.res(x)
        return x
 
 
class SeismicPickerUNetDet(nn.Module):
    """
    Identical to Step 2.
    The improvement in this step comes entirely from the training data,
    not the model. This is intentional and important — it demonstrates that
    data diversity is often more impactful than architectural complexity.
    """
 
    def __init__(
        self,
        in_channels: int   = 3,
        base_ch: int       = 32,
        lstm_hidden: int   = 128,
        lstm_layers: int   = 2,
        dropout: float     = 0.2,
    ):
        super().__init__()
 
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(),
        )
        self.enc1 = EncoderBlock(base_ch,     base_ch * 2, dilation=1)
        self.enc2 = EncoderBlock(base_ch * 2, base_ch * 4, dilation=2)
        self.enc3 = EncoderBlock(base_ch * 4, base_ch * 8, dilation=4)
 
        bottleneck_ch = base_ch * 8
 
        self.lstm = nn.LSTM(
            input_size=bottleneck_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_dropout = nn.Dropout(dropout)
        self.lstm_proj = nn.Sequential(
            nn.Conv1d(lstm_hidden * 2, bottleneck_ch, kernel_size=1),
            nn.BatchNorm1d(bottleneck_ch),
            nn.ReLU(),
        )
 
        self.dec1 = DecoderBlock(bottleneck_ch, base_ch * 4, base_ch * 4)
        self.dec2 = DecoderBlock(base_ch * 4,  base_ch * 2, base_ch * 2)
        self.dec3 = DecoderBlock(base_ch * 2,  base_ch,     base_ch)
        self.final_conv = nn.Sequential(
            nn.Conv1d(base_ch, base_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(),
        )
 
        self.head_p = nn.Conv1d(base_ch, 1, kernel_size=1)
        self.head_s = nn.Conv1d(base_ch, 1, kernel_size=1)
 
        self.det_head = nn.Sequential(
            nn.Linear(base_ch, base_ch // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(base_ch // 2, 1),
        )
 
    def forward(self, x: torch.Tensor):
        B, C, L = x.shape
        x = self.stem(x)
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
 
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.lstm_dropout(x)
        x = x.permute(0, 2, 1)
        x = self.lstm_proj(x)
 
        x = self.dec1(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec3(x, skip1)
        x = self.final_conv(x)
 
        prob_p    = torch.sigmoid(self.head_p(x).squeeze(1))
        prob_s    = torch.sigmoid(self.head_s(x).squeeze(1))
        pooled    = x.mean(dim=-1)
        det_score = torch.sigmoid(self.det_head(pooled).squeeze(1))
 
        return prob_p, prob_s, det_score
 
 
# ──────────────────────────────────────────────────────────────────────────────
# LOSS  (identical to Step 2)
# ──────────────────────────────────────────────────────────────────────────────
 
def phase_loss(pred, target, pos_weight_factor=10.0):
    pos_weight = torch.tensor(pos_weight_factor, device=pred.device)
    return F.binary_cross_entropy(pred, target, weight=(1 + (pos_weight - 1) * target))
 
 
def detection_loss(det_score, y_det):
    det_label = (y_det.squeeze(1).max(dim=-1).values > 0.5).float()
    return F.binary_cross_entropy(det_score, det_label)
 
 
def combined_loss(prob_p, prob_s, det_score, label_p, label_s, y_det,
                  w_phase=1.0, w_det=0.5):
    return w_phase * (phase_loss(prob_p, label_p) + phase_loss(prob_s, label_s)) \
         + w_det   * detection_loss(det_score, y_det)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# METRICS  (identical to Step 2)
# ──────────────────────────────────────────────────────────────────────────────
 
def pick_residuals(prob, label, threshold=0.3, tolerance=50):
    pred_pick = prob.argmax(dim=1)
    pred_max  = prob.max(dim=1).values
    true_pick = label.argmax(dim=1)
    has_pick  = label.max(dim=1).values > 0.1
    residuals = []
    for i in range(len(prob)):
        if not has_pick[i] or pred_max[i] < threshold:
            continue
        res = abs(pred_pick[i].item() - true_pick[i].item())
        if res <= tolerance:
            residuals.append(res)
    if not residuals:
        return float("nan"), 0
    return float(np.mean(residuals)), len(residuals)
 
 
def detection_accuracy(det_score, y_det, threshold=0.5):
    det_label  = (y_det.squeeze(1).max(dim=-1).values > 0.5).float()
    pred_label = (det_score > threshold).float()
    return (pred_label == det_label).float().mean().item()
 
 
# ──────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP  (minimal changes from Step 2)
# ──────────────────────────────────────────────────────────────────────────────
 
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int        = 50,
    learning_rate: float = 1e-3,
    device: str          = "cuda",
    checkpoint_dir: str  = "checkpoints/step3_multidataset",
):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    model = model.to(device)
 
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer, max_lr=learning_rate,
        steps_per_epoch=len(train_loader), epochs=n_epochs,
        pct_start=0.3, anneal_strategy="cos",
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    best_val_loss = float("inf")
 
    for epoch in range(1, n_epochs + 1):
 
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
 
        for batch in train_loader:
            waveform = batch["X"].to(device)
            label_p  = batch["y_p"].squeeze(1).to(device)
            label_s  = batch["y_s"].squeeze(1).to(device)
            y_det    = batch["y_det"].to(device)
 
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                prob_p, prob_s, det_score = model(waveform)
                loss = combined_loss(prob_p, prob_s, det_score, label_p, label_s, y_det)
 
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()
 
        train_loss /= len(train_loader)
 
        # ── Validate ──────────────────────────────────────────────────────
        # Validate on STEAD dev only — keep this consistent across all steps
        # so metrics are comparable to your Step 1 and Step 2 baselines.
        # You can add a separate INSTANCE validation loop if needed.
        model.eval()
        val_loss = 0.0
        p_res, s_res, det_accs = [], [], []
 
        with torch.no_grad():
            for batch in val_loader:
                waveform = batch["X"].to(device)
                label_p  = batch["y_p"].squeeze(1).to(device)
                label_s  = batch["y_s"].squeeze(1).to(device)
                y_det    = batch["y_det"].to(device)
 
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    prob_p, prob_s, det_score = model(waveform)
                    val_loss += combined_loss(
                        prob_p, prob_s, det_score, label_p, label_s, y_det
                    ).item()
 
                mp, _ = pick_residuals(prob_p.cpu(), label_p.cpu())
                ms, _ = pick_residuals(prob_s.cpu(), label_s.cpu())
                da    = detection_accuracy(det_score.cpu(), y_det.cpu())
                if not np.isnan(mp): p_res.append(mp)
                if not np.isnan(ms): s_res.append(ms)
                det_accs.append(da)
 
        val_loss /= len(val_loader)
        print(
            f"Epoch {epoch:03d}/{n_epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"P_res={np.mean(p_res) if p_res else float('nan'):.1f}samp | "
            f"S_res={np.mean(s_res) if s_res else float('nan'):.1f}samp | "
            f"det_acc={np.mean(det_accs):.3f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )
 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, f"{checkpoint_dir}/best_model.pt")
 
    return model
 
 
# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE  (identical to Step 2)
# ──────────────────────────────────────────────────────────────────────────────
 
@torch.no_grad()
def predict_single(model: nn.Module, waveform: np.ndarray, device: str = "cuda") -> dict:
    model.eval()
    x = torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0).to(device)
    prob_p, prob_s, det_score = model(x)
    prob_p    = prob_p[0].cpu().numpy()
    prob_s    = prob_s[0].cpu().numpy()
    det_score = det_score[0].item()
    p_sample  = int(prob_p.argmax())
    s_sample  = int(prob_s.argmax())
    return {
        "prob_p": prob_p, "prob_s": prob_s,
        "p_sample": p_sample, "s_sample": s_sample,
        "p_confidence": float(prob_p.max()),
        "s_confidence": float(prob_s.max()),
        "detection_score": det_score,
        "sp_interval_s": (s_sample - p_sample) / 100.0,
    }
 
 
# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    N_EPOCHS   = 50
    LR         = 1e-3
 
    # ── Multi-dataset train loader ─────────────────────────────────────────
    # Each dict is passed as kwargs to SeisBenchPipelineWrapper.
    # Add more datasets here (e.g., GEOFON, TXED) by appending to the list.
    train_loader = build_multi_dataset_loader(
        dataset_configs=[
            dict(dataset_name="STEAD",    split="train", model_type="eqtransformer",
                 transformation_sigma=10),
            dict(dataset_name="INSTANCE", split="train", model_type="eqtransformer",
                 transformation_sigma=10),
        ],
        batch_size=BATCH_SIZE,
        num_workers=4,
        equal_mixing=True,    # 50/50 regardless of dataset size
    )
 
    # ── Val loader — STEAD only ────────────────────────────────────────────
    # WHY validate on STEAD only: you want a fixed, consistent benchmark
    # across all three steps so your metrics are directly comparable.
    # Run a separate evaluation pass on INSTANCE dev after training to
    # measure cross-dataset generalization independently.
    val_pipe = SeisBenchPipelineWrapper(
        dataset_name="STEAD", split="dev", model_type="eqtransformer",
        transformation_sigma=10,
    )
    val_loader = val_pipe.get_dataloader(batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
 
    # ── Model (same as Step 2) ─────────────────────────────────────────────
    model = SeismicPickerUNetDet(
        in_channels=3,
        base_ch=32,
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.2,
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
 
    # ── Train ─────────────────────────────────────────────────────────────
    model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=N_EPOCHS,
        learning_rate=LR,
        device=DEVICE,
        checkpoint_dir="checkpoints/step3_multidataset",
    )
 
    # ── Cross-dataset evaluation ───────────────────────────────────────────
    # Run a separate pass on INSTANCE dev to measure true generalization.
    print("\n--- Cross-dataset evaluation on INSTANCE dev ---")
    instance_val_pipe = SeisBenchPipelineWrapper(
        dataset_name="INSTANCE", split="dev", model_type="eqtransformer",
        transformation_sigma=10,
    )
    instance_val_loader = instance_val_pipe.get_dataloader(
        batch_size=BATCH_SIZE, num_workers=4, shuffle=False,
    )
 
    model.eval()
    p_res_inst, s_res_inst = [], []
    with torch.no_grad():
        for batch in instance_val_loader:
            waveform = batch["X"].to(DEVICE)
            label_p  = batch["y_p"].squeeze(1).to(DEVICE)
            label_s  = batch["y_s"].squeeze(1).to(DEVICE)
            prob_p, prob_s, _ = model(waveform)
            mp, _ = pick_residuals(prob_p.cpu(), label_p.cpu())
            ms, _ = pick_residuals(prob_s.cpu(), label_s.cpu())
            if not np.isnan(mp): p_res_inst.append(mp)
            if not np.isnan(ms): s_res_inst.append(ms)
 
    print(
        f"INSTANCE dev | "
        f"P_res={np.mean(p_res_inst) if p_res_inst else float('nan'):.1f}samp | "
        f"S_res={np.mean(s_res_inst) if s_res_inst else float('nan'):.1f}samp"
    )
 
    # ── Sanity check ──────────────────────────────────────────────────────
    dummy = np.random.randn(3, 6000).astype(np.float32)
    out   = predict_single(model, dummy, device=DEVICE)
    print(
        f"\nSanity check | detection={out['detection_score']:.3f} | "
        f"P: sample={out['p_sample']}  conf={out['p_confidence']:.3f} | "
        f"S: sample={out['s_sample']}  conf={out['s_confidence']:.3f} | "
        f"S-P={out['sp_interval_s']:.2f}s"
    )