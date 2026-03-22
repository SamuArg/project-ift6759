"""
STEP 1 — CNN U-Net encoder/decoder + BiLSTM bottleneck
=======================================================
Change from baseline: the flat CNN → BiLSTM → heads architecture is replaced
by a proper U-Net with skip connections. The BiLSTM sits at the bottleneck.
 
WHY U-Net skip connections are the single biggest architectural win:
    The baseline model downsamples 6000 → 1500, runs BiLSTM, then naively
    interpolates back to 6000. That interpolation has no access to the original
    fine-grained waveform features — it is literally guessing where the sharp
    probability peak should be based on smoothed bottleneck features alone.
 
    U-Net skip connections route the encoder feature maps at each resolution
    directly to the corresponding decoder stage. The decoder then learns to
    COMBINE coarse semantic context (from the bottleneck) with fine temporal
    detail (from the skip). This is exactly what makes the output probability
    curve sharp and precisely localized rather than blurry.
 
    PhaseNet (Zhu & Beroza 2019) is literally a 1D U-Net and it remains
    competitive with EQTransformer on STEAD purely because of this property.
    The skip connections are not a minor tweak — they are the core mechanism
    that enables sub-sample pick precision.
 
Architecture diagram:
                        skip_3 (B,128,750) ─────────────────────┐
                        skip_2 (B, 64,1500) ──────────────┐     │
                        skip_1 (B, 32,3000) ─────────┐    │     │
                                                      │    │     │
    Input (B,3,6000)                                  │    │     │
    → Stem → ResBlocks (B,32,6000)                    │    │     │
    → Down1             (B, 64,3000) → skip_1 ────────┤    │     │
    → Down2             (B,128,1500) → skip_2 ─────────┤    │     │
    → Down3             (B,256, 750) → skip_3 ──────────┤    │     │
    → BiLSTM bottleneck (B,256, 750)                   │    │     │
    → Up1  + concat(skip_3) → (B,128, 750) ────────────┘    │     │
    → Up2  + concat(skip_2) → (B, 64,1500) ─────────────────┘     │
    → Up3  + concat(skip_1) → (B, 32,3000) ───────────────────────┘
    → Final upsample        → (B, 32,6000)
    → head_p, head_s        → sigmoid → (B, 6000)
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
# DATA PIPELINE 
# ──────────────────────────────────────────────────────────────────────────────
 
class SeisBenchPipelineWrapper:
    def __init__(self, dataset_name="STEAD", split="train", model_type="eqtransformer",
                 component_order="ZNE", max_distance=None, transformation_shape="gaussian",
                 transformation_sigma=10, dataset_fraction=1.0):
        self.dataset_name        = dataset_name.upper()
        self.split               = split.lower()
        self.model_type          = model_type.lower()
        self.component_order     = component_order
        self.max_distance        = max_distance
        self.transformation_shape = transformation_shape
        self.transformation_sigma = transformation_sigma
        self.dataset_fraction    = dataset_fraction
 
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
        ]
        self.generator.add_augmentations(augmentations)
 
    def get_dataloader(self, batch_size=32, num_workers=4, shuffle=True):
        return DataLoader(self.generator, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# MODEL BUILDING BLOCKS
# ──────────────────────────────────────────────────────────────────────────────
 
class ResidualConvBlock(nn.Module):
    """
    Two Conv1d layers with BatchNorm and a residual skip.
    Used in both the encoder and decoder paths.
 
    WHY dilation in the encoder: dilated convolutions grow the receptive field
    exponentially without adding parameters. dilation=1,2,4 after 3 blocks
    covers ~0.5s of context, enough to see the P-wave onset shape.
 
    WHY no dilation in the decoder: the decoder needs spatially precise
    features to reconstruct the sharp probability peak. Dilation smears
    spatial precision — exactly what you want to avoid when upsampling.
    """
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
    """
    One U-Net encoder stage:
        ResidualConvBlock (at current resolution) → strided Conv1d (downsample ×2)
 
    Returns BOTH the pre-downsample features (the skip connection) AND the
    downsampled features (passed to the next stage).
 
    WHY return features before downsampling as the skip:
        The skip connection must carry full-resolution temporal detail.
        If we skipped after downsampling we'd be passing coarser features,
        defeating the purpose of the skip.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        self.res   = ResidualConvBlock(in_ch, kernel_size=kernel_size, dilation=dilation)
        # Strided conv: learned downsampling, halves sequence length
        self.down  = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )
 
    def forward(self, x):
        x    = self.res(x)    # (B, in_ch, L)    ← skip connection captured here
        skip = x
        x    = self.down(x)   # (B, out_ch, L/2) ← passed to next encoder stage
        return x, skip
 
 
class DecoderBlock(nn.Module):
    """
    One U-Net decoder stage:
        Upsample ×2 → concat skip → Conv1d to merge → ResidualConvBlock
 
    WHY concat then conv, not add:
        Concatenation preserves ALL information from both paths (bottleneck
        context + encoder detail) and lets the conv decide how to weight each.
        Addition forces the two paths to live in the same feature space, which
        is a stronger constraint and loses information.
 
    WHY upsample then concat (not concat then upsample):
        We need the upsampled tensor to match the skip's spatial dimension
        before concatenation. The merge conv operates at full resolution.
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, kernel_size: int = 7):
        super().__init__()
        # After concat: channels = in_ch (from below) + skip_ch (from encoder)
        self.merge = nn.Sequential(
            nn.Conv1d(in_ch + skip_ch, out_ch, kernel_size=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )
        self.res = ResidualConvBlock(out_ch, kernel_size=kernel_size, dilation=1)
 
    def forward(self, x, skip):
        # x:    (B, in_ch,   L)
        # skip: (B, skip_ch, L*2) ← from matching encoder stage
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, skip], dim=1)   # (B, in_ch+skip_ch, L*2)
        x = self.merge(x)                 # (B, out_ch, L*2)
        x = self.res(x)                   # (B, out_ch, L*2)
        return x
 
 
# ──────────────────────────────────────────────────────────────────────────────
# FULL MODEL
# ──────────────────────────────────────────────────────────────────────────────
 
class SeismicPickerUNet(nn.Module):
    """
    U-Net CNN + BiLSTM bottleneck seismic phase picker.
 
    Channel progression (base_ch=32):
        Encoder:  3 → 32 → 64 → 128 → 256  (4 stages, each halves seq length)
        Bottleneck: BiLSTM at (B, 256, 750) → processes (B, 750, 256) → back to (B, 256, 750)
        Decoder:  256+128→128 → 128+64→64 → 64+32→32 → upsample to 6000
        Heads: two independent 1×1 convs → sigmoid
 
    Sequence lengths at each stage (input=6000):
        After enc1: 3000   skip_1: (B, 32,  3000)
        After enc2: 1500   skip_2: (B, 64,  1500)
        After enc3:  750   skip_3: (B, 128,  750)
        Bottleneck:  750          (B, 256,  750)  ← BiLSTM here
        After dec1: 1500   (merged with skip_3)
        After dec2: 3000   (merged with skip_2)
        After dec3: 6000   (merged with skip_1)
    """
 
    def __init__(
        self,
        in_channels: int   = 3,
        base_ch: int       = 32,    # channel width; doubles at each encoder stage
        lstm_hidden: int   = 128,   # per-direction hidden size in BiLSTM
        lstm_layers: int   = 2,
        dropout: float     = 0.2,
    ):
        super().__init__()
 
        # ── Stem ──────────────────────────────────────────────────────────
        # Projects 3 raw channels → base_ch WITHOUT changing sequence length.
        # kernel_size=7 → 70ms receptive field at 100Hz.
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(),
        )
 
        # ── Encoder ───────────────────────────────────────────────────────
        # Increasing dilation per stage grows receptive field without
        # adding parameters. By enc3 the model sees ~0.5s of context.
        self.enc1 = EncoderBlock(base_ch,     base_ch * 2,  dilation=1)   # 32→64,  6000→3000
        self.enc2 = EncoderBlock(base_ch * 2, base_ch * 4,  dilation=2)   # 64→128, 3000→1500
        self.enc3 = EncoderBlock(base_ch * 4, base_ch * 8,  dilation=4)   # 128→256, 1500→750
 
        # ── Bottleneck projection ──────────────────────────────────────────
        # After enc3 we have (B, 256, 750). The BiLSTM input_size must
        # match the channel dim (256 here).
        bottleneck_ch = base_ch * 8   # 256
 
        # ── BiLSTM bottleneck ──────────────────────────────────────────────
        # WHY BiLSTM at the bottleneck and not across the full 6000 samples:
        #   At 750 timesteps the BiLSTM is 8x cheaper than at 6000.
        #   The bottleneck features are maximally abstract — perfect for the
        #   LSTM to model long-range event structure (e.g., the temporal
        #   relationship between P and S arrivals, which can be seconds apart).
        #   Fine-grained waveform detail is handled by the CNN; the LSTM
        #   handles the sequential, event-level context.
        self.lstm = nn.LSTM(
            input_size=bottleneck_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_dropout = nn.Dropout(dropout)
 
        lstm_out = lstm_hidden * 2   # 256 (bidirectional)
 
        # Project LSTM output back to bottleneck_ch so decoder channel math is clean
        # WHY: LSTM output is lstm_hidden*2=256; if lstm_hidden != bottleneck_ch/2
        # this projection handles the mismatch without special-casing decoder dims.
        self.lstm_proj = nn.Sequential(
            nn.Conv1d(lstm_out, bottleneck_ch, kernel_size=1),
            nn.BatchNorm1d(bottleneck_ch),
            nn.ReLU(),
        )
 
        # ── Decoder ───────────────────────────────────────────────────────
        # Each DecoderBlock receives:
        #   - features from the stage below (in_ch)
        #   - the skip connection from the matching encoder stage (skip_ch)
        # Output channels (out_ch) are halved at each decoder stage,
        # symmetric to the encoder.
        self.dec1 = DecoderBlock(bottleneck_ch, base_ch * 4, base_ch * 4)  # 256+128→128
        self.dec2 = DecoderBlock(base_ch * 4,  base_ch * 2, base_ch * 2)  # 128+64 →64
        self.dec3 = DecoderBlock(base_ch * 2,  base_ch,     base_ch)      # 64+32  →32
 
        # Final upsample: 3000 → 6000 (match original input length)
        # The last decoder stage outputs at 3000 (matching skip_1 at 3000).
        # We still need one more 2x upsample to reach 6000.
        self.final_conv = nn.Sequential(
            nn.Conv1d(base_ch, base_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(),
        )
 
        # ── Output heads ──────────────────────────────────────────────────
        # Two independent 1×1 convolutions — one per phase.
        # WHY independent: P and S picking are related tasks (shared backbone)
        # but require distinct decision functions (separate heads).
        # 1×1 conv = linear transform per timestep; all temporal context
        # has already been captured by the encoder + BiLSTM + decoder.
        self.head_p = nn.Conv1d(base_ch, 1, kernel_size=1)
        self.head_s = nn.Conv1d(base_ch, 1, kernel_size=1)
 
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, 6000) — normalized 3-component waveform
 
        Returns:
            prob_p: (B, 6000) — P-wave arrival probability per sample
            prob_s: (B, 6000) — S-wave arrival probability per sample
        """
        B, C, L = x.shape
 
        # Stem
        x = self.stem(x)                        # (B, 32, 6000)
 
        # Encoder (each stage returns downsampled features + skip)
        x, skip1 = self.enc1(x)                 # x:(B,64,3000)  skip1:(B,32,6000)
        x, skip2 = self.enc2(x)                 # x:(B,128,1500) skip2:(B,64,3000)
        x, skip3 = self.enc3(x)                 # x:(B,256,750)  skip3:(B,128,1500)
 
        # BiLSTM bottleneck
        x = x.permute(0, 2, 1)                  # (B, 750, 256)
        x, _ = self.lstm(x)                      # (B, 750, 256)
        x = self.lstm_dropout(x)
        x = x.permute(0, 2, 1)                  # (B, 256, 750)
        x = self.lstm_proj(x)                    # (B, 256, 750)
 
        # Decoder (each stage upsamples and merges its matching skip)
        x = self.dec1(x, skip3)                  # (B, 128, 1500)
        x = self.dec2(x, skip2)                  # (B, 64,  3000)
        x = self.dec3(x, skip1)                  # (B, 32,  6000)
 
        # Final conv at full resolution
        x = self.final_conv(x)                   # (B, 32, 6000)
 
        # Heads + sigmoid
        prob_p = torch.sigmoid(self.head_p(x).squeeze(1))   # (B, 6000)
        prob_s = torch.sigmoid(self.head_s(x).squeeze(1))   # (B, 6000)
 
        return prob_p, prob_s
 
 
# ──────────────────────────────────────────────────────────────────────────────
# LOSS
# ──────────────────────────────────────────────────────────────────────────────
 
def phase_loss(pred: torch.Tensor, target: torch.Tensor,
               pos_weight_factor: float = 10.0) -> torch.Tensor:
    """
    Weighted BCE.
 
    WHY BCE not MSE: target is a Gaussian probability distribution.
    BCE is the correct divergence between two distributions.
    MSE penalises large deviations quadratically → overly smooth, flat peaks.
 
    WHY positive class weighting: arrivals occupy ~20/6000 samples (0.3%).
    Without reweighting the model collapses to predicting all-zero everywhere
    and achieves 99.7% "accuracy" while picking nothing.
    """
    pos_weight = torch.tensor(pos_weight_factor, device=pred.device)
    return F.binary_cross_entropy(pred, target, weight=(1 + (pos_weight - 1) * target))
 
 
# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────
 
def pick_residuals(prob: torch.Tensor, label: torch.Tensor,
                   threshold: float = 0.3, tolerance: int = 50):
    """
    Mean absolute residual in samples between predicted and true pick.
    tolerance=50 → 0.5s association window.
    Only counts traces where a ground-truth pick exists AND model is confident.
    """
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
    checkpoint_dir: str  = "checkpoints/step1_unet",
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
 
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                prob_p, prob_s = model(waveform)
                loss = phase_loss(prob_p, label_p) + phase_loss(prob_s, label_s)
 
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
        p_res, s_res = [], []
 
        with torch.no_grad():
            for batch in val_loader:
                waveform = batch["X"].to(device)
                label_p  = batch["y_p"].squeeze(1).to(device)
                label_s  = batch["y_s"].squeeze(1).to(device)
 
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    prob_p, prob_s = model(waveform)
                    val_loss += (phase_loss(prob_p, label_p) + phase_loss(prob_s, label_s)).item()
 
                mp, _ = pick_residuals(prob_p.cpu(), label_p.cpu())
                ms, _ = pick_residuals(prob_s.cpu(), label_s.cpu())
                if not np.isnan(mp): p_res.append(mp)
                if not np.isnan(ms): s_res.append(ms)
 
        val_loss /= len(val_loader)
        print(
            f"Epoch {epoch:03d}/{n_epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"P_res={np.mean(p_res) if p_res else float('nan'):.1f}samp | "
            f"S_res={np.mean(s_res) if s_res else float('nan'):.1f}samp | "
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
    prob_p, prob_s = model(x)
    prob_p = prob_p[0].cpu().numpy()
    prob_s = prob_s[0].cpu().numpy()
    p_sample = int(prob_p.argmax())
    s_sample = int(prob_s.argmax())
    return {
        "prob_p": prob_p, "prob_s": prob_s,
        "p_sample": p_sample, "s_sample": s_sample,
        "p_confidence": float(prob_p.max()),
        "s_confidence": float(prob_s.max()),
        "sp_interval_s": (s_sample - p_sample) / 100.0,
    }
 
 
# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32    # U-Net uses more memory than the flat baseline; reduce if OOM
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
 
    model = SeismicPickerUNet(
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
        checkpoint_dir="checkpoints/step1_unet",
    )
 
    dummy = np.random.randn(3, 6000).astype(np.float32)
    out   = predict_single(model, dummy, device=DEVICE)
    print(
        f"P: sample={out['p_sample']}  conf={out['p_confidence']:.3f} | "
        f"S: sample={out['s_sample']}  conf={out['s_confidence']:.3f} | "
        f"S-P={out['sp_interval_s']:.2f}s"
    )
 