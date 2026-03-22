"""
STEP 2 — U-Net + BiLSTM + noise/detection head
===============================================
Builds directly on Step 1. The only addition is a third output head:
a noise/detection classifier that predicts whether the window contains
an earthquake at all.
 
WHY add a detection head:
    STEAD contains ~30% noise traces. INSTANCE is closer to 50% noise.
    When trained WITHOUT a detection head, the model sees noise traces and
    tries to find a P/S arrival anyway — the positive class weighting in the
    BCE loss pressures it to pick SOMETHING. The result is false picks on
    high-amplitude noise that happens to look vaguely like a P-wave onset.
 
    Adding an explicit detection head changes the problem:
        - The shared backbone must now learn features that distinguish
          earthquake waveforms from noise waveforms, not just locate phases.
        - On noise traces the detection head learns to suppress the P/S heads.
        - The model develops a more physically meaningful internal
          representation because it cannot cheat — it has to answer
          "is there an earthquake here?" before answering "where?"
 
    This is directly analogous to what EQTransformer does with its detector
    head, and it is one of the main reasons EQTransformer outperforms pure
    PhaseNet on noisy data.
 
What changes vs Step 1:
    1. SeisBenchPipelineWrapper adds DetectionLabeller for a "y_det" label.
    2. SeismicPickerUNet gains a detection head (global average pool → linear).
    3. Loss function gains a detection BCE term.
    4. Training loop unpacks y_det and passes it through.
    5. Validation metrics add detection accuracy.
    6. predict_single returns detection score.
 
Architecture addition (detection head):
    Shared decoder output (B, 32, 6000)
    → Global Average Pool across time → (B, 32)
    → Linear(32, 1) → sigmoid → detection score scalar ∈ [0, 1]
 
    WHY global average pool:
        Detection is a window-level binary decision ("earthquake or not").
        It should not be sensitive to where in the window the earthquake is,
        so pooling across time is the correct operation.
        A simple linear layer on top of the pooled features is sufficient —
        the decoder has already done the heavy feature extraction.
"""
 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
 
import seisbench.data as sbd
import seisbench.generate as sbg
 
 
# ──────────────────────────────────────────────────────────────────────────────
# DATA PIPELINE  (adds DetectionLabeller vs Step 1)
# ──────────────────────────────────────────────────────────────────────────────
 
class SeisBenchPipelineWrapper:
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
            # ── NEW in Step 2 ──────────────────────────────────────────────
            # DetectionLabeller outputs a window-level binary label:
            #   1.0 if the window contains a P or S arrival, 0.0 if noise.
            # The label is a time-series (shape matches X), but we pool
            # it to a scalar in the model — see detection head below.
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
# MODEL BUILDING BLOCKS  (identical to Step 1)
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
 
 
# ──────────────────────────────────────────────────────────────────────────────
# FULL MODEL  (Step 1 + detection head)
# ──────────────────────────────────────────────────────────────────────────────
 
class SeismicPickerUNetDet(nn.Module):
    """
    U-Net + BiLSTM bottleneck + detection head.
 
    Three outputs:
        prob_p   (B, 6000)  — P-wave arrival probability per sample
        prob_s   (B, 6000)  — S-wave arrival probability per sample
        det_score (B,)      — probability that the window contains an earthquake
 
    The detection head branches off the DECODER output (B, base_ch, 6000),
    AFTER the skip connections have merged fine and coarse features.
 
    WHY branch from the decoder and not the bottleneck:
        The bottleneck sees compressed, abstract features. The decoder output
        has already merged high-resolution encoder detail — it is richer.
        A detection head on the decoder is therefore a stronger classifier.
        It also means the detection loss backpropagates through the full
        decoder path, regularizing ALL layers to learn earthquake-relevant
        features, not just the bottleneck.
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
 
        # ── Stem + Encoder (identical to Step 1) ──────────────────────────
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(),
        )
        self.enc1 = EncoderBlock(base_ch,     base_ch * 2, dilation=1)
        self.enc2 = EncoderBlock(base_ch * 2, base_ch * 4, dilation=2)
        self.enc3 = EncoderBlock(base_ch * 4, base_ch * 8, dilation=4)
 
        bottleneck_ch = base_ch * 8
 
        # ── BiLSTM bottleneck (identical to Step 1) ────────────────────────
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
 
        # ── Decoder (identical to Step 1) ──────────────────────────────────
        self.dec1 = DecoderBlock(bottleneck_ch, base_ch * 4, base_ch * 4)
        self.dec2 = DecoderBlock(base_ch * 4,  base_ch * 2, base_ch * 2)
        self.dec3 = DecoderBlock(base_ch * 2,  base_ch,     base_ch)
        self.final_conv = nn.Sequential(
            nn.Conv1d(base_ch, base_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(),
        )
 
        # ── Phase heads (identical to Step 1) ──────────────────────────────
        self.head_p = nn.Conv1d(base_ch, 1, kernel_size=1)
        self.head_s = nn.Conv1d(base_ch, 1, kernel_size=1)
 
        # ── Detection head (NEW in Step 2) ─────────────────────────────────
        # Global average pool collapses (B, base_ch, 6000) → (B, base_ch).
        # A small MLP then maps to a single detection probability.
        # WHY two linear layers with ReLU (not just one):
        #   The intermediate layer (base_ch → base_ch // 2) gives the
        #   detection head a small amount of non-linear capacity to
        #   combine channel features before the final binary decision.
        #   One linear layer would be a purely linear classifier on the
        #   pooled features, which is too restrictive.
        self.det_head = nn.Sequential(
            nn.Linear(base_ch, base_ch // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(base_ch // 2, 1),
        )
 
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, 6000)
 
        Returns:
            prob_p:    (B, 6000)
            prob_s:    (B, 6000)
            det_score: (B,)      ← NEW
        """
        B, C, L = x.shape
 
        # Encoder
        x = self.stem(x)
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
 
        # BiLSTM bottleneck
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.lstm_dropout(x)
        x = x.permute(0, 2, 1)
        x = self.lstm_proj(x)
 
        # Decoder
        x = self.dec1(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec3(x, skip1)
        x = self.final_conv(x)           # (B, base_ch, 6000)
 
        # Phase outputs
        prob_p = torch.sigmoid(self.head_p(x).squeeze(1))   # (B, 6000)
        prob_s = torch.sigmoid(self.head_s(x).squeeze(1))   # (B, 6000)
 
        # Detection output
        # Global average pool: (B, base_ch, 6000) → (B, base_ch)
        pooled    = x.mean(dim=-1)                           # (B, base_ch)
        det_score = torch.sigmoid(self.det_head(pooled).squeeze(1))  # (B,)
 
        return prob_p, prob_s, det_score
 
 
# ──────────────────────────────────────────────────────────────────────────────
# LOSS
# ──────────────────────────────────────────────────────────────────────────────
 
def phase_loss(pred: torch.Tensor, target: torch.Tensor,
               pos_weight_factor: float = 10.0) -> torch.Tensor:
    pos_weight = torch.tensor(pos_weight_factor, device=pred.device)
    return F.binary_cross_entropy(pred, target, weight=(1 + (pos_weight - 1) * target))
 
 
def detection_loss(det_score: torch.Tensor, y_det: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy for the window-level detection task.
 
    y_det from SeisBench DetectionLabeller is a time-series (B, 1, 6000)
    where 1.0 indicates the sample is inside an earthquake window.
    We collapse it to a single scalar per trace by taking the max:
        max > 0.5  →  earthquake window  →  label = 1.0
        max ≤ 0.5  →  noise window       →  label = 0.0
 
    WHY max not mean: a trace is an earthquake trace if ANY sample has
    a phase arrival. The mean would dilute the label for short-duration events.
 
    WHY unweighted BCE here (unlike phase_loss):
        At the window level the class imbalance is much milder (~30-50% noise).
        Standard BCE is sufficient; heavy weighting would push the model to
        over-predict detections.
    """
    # y_det shape: (B, 1, 6000) → collapse to (B,)
    det_label = (y_det.squeeze(1).max(dim=-1).values > 0.5).float()
    return F.binary_cross_entropy(det_score, det_label)
 
 
def combined_loss(prob_p, prob_s, det_score, label_p, label_s, y_det,
                  w_phase: float = 1.0, w_det: float = 0.5) -> torch.Tensor:
    """
    Weighted sum of phase picking loss and detection loss.
 
    WHY w_det = 0.5 (not 1.0):
        Phase picking is the primary task. We don't want the detection signal
        to dominate and cause the model to sacrifice pick precision for better
        noise/earthquake discrimination.
        0.5 is a reasonable starting point; tune if detection recall is poor.
    """
    lp  = phase_loss(prob_p, label_p)
    ls  = phase_loss(prob_s, label_s)
    ld  = detection_loss(det_score, y_det)
    return w_phase * (lp + ls) + w_det * ld
 
 
# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────
 
def pick_residuals(prob: torch.Tensor, label: torch.Tensor,
                   threshold: float = 0.3, tolerance: int = 50):
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
 
 
def detection_accuracy(det_score: torch.Tensor, y_det: torch.Tensor,
                        threshold: float = 0.5) -> float:
    """Binary accuracy of the detection head."""
    det_label = (y_det.squeeze(1).max(dim=-1).values > 0.5).float()
    pred_label = (det_score > threshold).float()
    return (pred_label == det_label).float().mean().item()
 
 
# ──────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────
 
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int        = 50,
    learning_rate: float = 1e-3,
    device: str          = "cuda",
    checkpoint_dir: str  = "checkpoints/step2_det",
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
            y_det    = batch["y_det"].to(device)              # (B, 1, 6000)
 
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
# INFERENCE
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
        "detection_score": det_score,          # ← NEW: use as a confidence gate
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
 
    train_pipe = SeisBenchPipelineWrapper(
        dataset_name="STEAD", split="train", model_type="eqtransformer",
        transformation_sigma=10,
    )
    val_pipe = SeisBenchPipelineWrapper(
        dataset_name="STEAD", split="dev", model_type="eqtransformer",
        transformation_sigma=10,
    )
    train_loader = train_pipe.get_dataloader(batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
    val_loader   = val_pipe.get_dataloader(batch_size=BATCH_SIZE,   num_workers=4, shuffle=False)
 
    model = SeismicPickerUNetDet(
        in_channels=3,
        base_ch=32,
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.2,
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
 
    model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=N_EPOCHS,
        learning_rate=LR,
        device=DEVICE,
        checkpoint_dir="checkpoints/step2_det",
    )
 
    dummy = np.random.randn(3, 6000).astype(np.float32)
    out   = predict_single(model, dummy, device=DEVICE)
    print(
        f"detection={out['detection_score']:.3f} | "
        f"P: sample={out['p_sample']}  conf={out['p_confidence']:.3f} | "
        f"S: sample={out['s_sample']}  conf={out['s_confidence']:.3f} | "
        f"S-P={out['sp_interval_s']:.2f}s"
    )
 