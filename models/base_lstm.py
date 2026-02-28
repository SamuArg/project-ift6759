"""
Seismic Phase Picker: CNN + BiLSTM for P-wave and S-wave arrival detection
Designed for STEAD and INSTANCE datasets.

Architecture overview:
    Raw 3C waveform → Conv1D encoder → BiLSTM → dual sigmoid heads (P + S)

Output: two probability vectors of shape (batch, seq_len) — one per phase.
        Each value is the probability that the sample is the arrival onset.
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
# DATA PIPELINE (SeisBench)
# ──────────────────────────────────────────────────────────────────────────────

class SeisBenchPipelineWrapper:
    def __init__(self, dataset_name="STEAD", split="train", model_type="eqtransformer",
                 component_order="ZNE", max_distance=None, transformation_shape="gaussian",
                 transformation_sigma=10, dataset_fraction=1.0):
        self.dataset_name = dataset_name.upper()
        self.split = split.lower()
        self.model_type = model_type.lower()
        self.component_order = component_order
        self.max_distance = max_distance
        self.transformation_shape = transformation_shape
        self.transformation_sigma = transformation_sigma
        self.dataset_fraction = dataset_fraction

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
            dist_col = next((col for col in possible_cols if col in self.dataset.metadata.columns), None)
            if dist_col:
                mask = (
                    (self.dataset.metadata[dist_col] <= self.max_distance) |
                    (self.dataset.metadata[dist_col].isna())
                ).values
                self.dataset.filter(mask)

        if 0.0 < self.dataset_fraction < 1.0:
            total_events = len(self.dataset)
            num_samples = int(total_events * self.dataset_fraction)
            subsample_mask = np.zeros(total_events, dtype=bool)
            subsample_mask[np.random.choice(total_events, num_samples, replace=False)] = True
            self.dataset.filter(subsample_mask)

        self.generator = sbg.GenericGenerator(self.dataset)
        self._attach_pipeline()

    def _attach_pipeline(self):
        augmentations = []
        window_len = 6000 if self.model_type == "eqtransformer" else 3001

        p_col = "trace_p_arrival_sample"
        s_col = "trace_s_arrival_sample"
        if p_col not in self.dataset.metadata.columns:
            p_col = "trace_P_arrival_sample"
            s_col = "trace_S_arrival_sample"

        augmentations.extend([
            sbg.WindowAroundSample(
                metadata_keys=[p_col, s_col],
                selection="random",
                samples_before=window_len // 2,
                strategy="pad",
                windowlen=window_len
            ),
            sbg.ChangeDtype(np.float32),
            sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="std"),
            sbg.ProbabilisticLabeller(
                label_columns=[p_col],
                shape=self.transformation_shape,
                sigma=self.transformation_sigma,
                key=("X", "y_p"),
                dim=0
            ),
            sbg.ChangeDtype(np.float32, key="y_p"),
            sbg.ProbabilisticLabeller(
                label_columns=[s_col],
                shape=self.transformation_shape,
                sigma=self.transformation_sigma,
                key=("X", "y_s"),
                dim=0
            ),
            sbg.ChangeDtype(np.float32, key="y_s"),
            sbg.ChangeDtype(np.float32, key="X"),
        ])

        if self.model_type == "eqtransformer":
            augmentations.extend([
                sbg.DetectionLabeller(p_phases=[p_col], s_phases=[s_col], key=("X", "y_det")),
                sbg.ChangeDtype(np.float32, key="y_det"),
            ])

        self.generator.add_augmentations(augmentations)

    def get_dataloader(self, batch_size=32, num_workers=4, shuffle=True):
        return DataLoader(
            self.generator,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────────────────────

class ResidualConvBlock(nn.Module):
    """
    Two Conv1d layers with a skip connection and BatchNorm.

    WHY residual: skip connections let gradients flow directly through the
    network, avoiding vanishing gradients when stacking conv layers. The model
    learns incremental refinements rather than full transformations at each layer.
    WHY BatchNorm: stabilizes activations per channel per batch, reducing
    sensitivity to weight initialization and learning rate.
    """

    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2  # 'same' padding
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(channels)
        self.bn2   = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class SeismicPicker(nn.Module):
    """
    CNN + BiLSTM seismic phase picker.

    Architecture:
        (B, 3, 6000)
        → Stem Conv1d           : 3 → 64 channels, captures ~70ms local shape
        → 3x ResidualConvBlock  : dilations 1, 2, 4 — receptive field grows to ~0.5s
                                  without adding parameters (dilated = exponential reach)
        → 2x Strided Conv1d     : 6000 → 1500 (4x downsample)
                                  WHY: LSTM over 1500 steps is 4x faster than 6000,
                                  with no meaningful precision loss since only the
                                  final upsampled output needs full resolution.
        → 2-layer BiLSTM        : bidirectional so S-wave context can use post-P info;
                                  2 layers captures hierarchical temporal patterns
        → Dropout
        → Two Conv1d heads      : independent P and S heads — shared backbone for
                                  common features, separate heads for distinct decisions
        → Linear upsample       : 1500 → 6000 via interpolation (no checkerboard artifacts)
        → Sigmoid               : outputs are per-sample arrival probabilities

    ~2.3M parameters — intentionally small for fast first iteration.
    """

    def __init__(
        self,
        in_channels: int   = 3,
        base_channels: int = 64,
        lstm_hidden: int   = 128,
        lstm_layers: int   = 2,
        dropout: float     = 0.2,
    ):
        super().__init__()

        # Stem: project 3 raw channels → base_channels
        # kernel_size=7 → 70ms receptive field at 100Hz
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
        )

        # Dilated residual encoder: receptive field ~0.5s after 3 blocks
        self.encoder = nn.Sequential(
            ResidualConvBlock(base_channels, kernel_size=7, dilation=1),
            ResidualConvBlock(base_channels, kernel_size=7, dilation=2),
            ResidualConvBlock(base_channels, kernel_size=7, dilation=4),
        )

        # Downsample: 6000 → 3000 → 1500
        # WHY strided conv over MaxPool: learns the downsampling kernel
        self.downsample = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(),
            nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(),
        )

        # BiLSTM: captures long-range temporal context
        # WHY LSTM over GRU: separate cell state gives better gradient flow
        # over long sequences; safer default at 1500 timesteps.
        self.lstm = nn.LSTM(
            input_size=base_channels * 2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_dropout = nn.Dropout(dropout)

        lstm_out = lstm_hidden * 2  # bidirectional doubles hidden size

        # Independent heads per phase
        # WHY 1x1 conv (= linear per timestep): the LSTM already has full
        # temporal context; no wide kernel needed at this stage
        self.head_p = nn.Conv1d(lstm_out, 1, kernel_size=1)
        self.head_s = nn.Conv1d(lstm_out, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, 6000) — normalized 3-component waveform from SeisBench

        Returns:
            prob_p: (B, 6000) — P-wave arrival probability per sample
            prob_s: (B, 6000) — S-wave arrival probability per sample
        """
        B, C, L = x.shape

        x = self.stem(x)         # (B, 64,  6000)
        x = self.encoder(x)      # (B, 64,  6000)
        x = self.downsample(x)   # (B, 128, 1500)

        x = x.permute(0, 2, 1)          # (B, 1500, 128) — LSTM expects (B, L, H)
        x, _ = self.lstm(x)              # (B, 1500, 256)
        x = self.lstm_dropout(x)
        x = x.permute(0, 2, 1)          # (B, 256, 1500) — back to (B, H, L) for Conv1d

        logit_p = self.head_p(x).squeeze(1)   # (B, 1500)
        logit_s = self.head_s(x).squeeze(1)   # (B, 1500)

        # Upsample to original length
        # WHY interpolation not transposed conv: smooth probability curves,
        # no checkerboard artifacts, no extra parameters needed
        prob_p = torch.sigmoid(
            F.interpolate(logit_p.unsqueeze(1), size=L, mode="linear", align_corners=False).squeeze(1)
        )
        prob_s = torch.sigmoid(
            F.interpolate(logit_s.unsqueeze(1), size=L, mode="linear", align_corners=False).squeeze(1)
        )

        return prob_p, prob_s   # each (B, 6000)


# ──────────────────────────────────────────────────────────────────────────────
# LOSS
# ──────────────────────────────────────────────────────────────────────────────

def phase_loss(pred: torch.Tensor, target: torch.Tensor,
               pos_weight_factor: float = 10.0) -> torch.Tensor:
    """
    Weighted Binary Cross-Entropy.

    WHY BCE not MSE: target is a probability distribution (Gaussian-shaped),
    so BCE is the correct divergence. MSE over-penalises large deviations and
    produces overly conservative, flat predictions.

    WHY positive class weighting: P/S arrivals occupy ~20 samples out of 6000
    (0.3% positive). Without reweighting the model collapses to all-zero output,
    which looks like 99.7% accuracy but picks nothing.
    pos_weight_factor=10 is a starting point — raise if you miss too many picks,
    lower if you get too many false positives.
    """
    pos_weight = torch.tensor(pos_weight_factor, device=pred.device)
    return F.binary_cross_entropy(pred, target, weight=(1 + (pos_weight - 1) * target))


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────

def pick_residuals(prob: torch.Tensor, label: torch.Tensor,
                   threshold: float = 0.3, tolerance: int = 50):
    """
    Mean absolute residual (in samples) between predicted and true pick,
    for traces where both a ground-truth pick and a confident prediction exist.

    WHY this over val loss: loss is an aggregate training signal.
    Pick residual directly answers "how many ms off is my pick?" —
    interpretable for seismologists and comparable to published benchmarks.

    tolerance=50 → 0.5s window to count a detection as a match.
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
    checkpoint_dir: str  = "checkpoints",
):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    model = model.to(device)

    # WHY AdamW: weight decay is applied directly to weights, not through the
    # gradient — gives cleaner regularisation than vanilla Adam.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # WHY OneCycleLR: ramp-up then cosine annealing consistently reaches good
    # solutions faster than fixed LR or StepLR. Steps per batch, not per epoch.
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=n_epochs,
        pct_start=0.3,
        anneal_strategy="cos",
    )

    # WHY AMP: halves VRAM usage and speeds up matmuls via float16 forward/backward,
    # float32 optimizer steps. Safe for this architecture.
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_val_loss = float("inf")

    for epoch in range(1, n_epochs + 1):

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            waveform = batch["X"].to(device)               # (B, 3, 6000)
            label_p  = batch["y_p"].squeeze(1).to(device)  # (B, 1, 6000) → (B, 6000)
            label_s  = batch["y_s"].squeeze(1).to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                prob_p, prob_s = model(waveform)
                loss = phase_loss(prob_p, label_p) + phase_loss(prob_s, label_s)

            scaler.scale(loss).backward()

            # WHY grad clipping: LSTMs are prone to gradient explosions early
            # in training. norm=1.0 keeps updates bounded without killing signal.
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
        p_residuals, s_residuals = [], []

        with torch.no_grad():
            for batch in val_loader:
                waveform = batch["X"].to(device)
                label_p  = batch["y_p"].squeeze(1).to(device)
                label_s  = batch["y_s"].squeeze(1).to(device)

                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    prob_p, prob_s = model(waveform)
                    val_loss += (phase_loss(prob_p, label_p) + phase_loss(prob_s, label_s)).item()

                mean_p, _ = pick_residuals(prob_p.cpu(), label_p.cpu())
                mean_s, _ = pick_residuals(prob_s.cpu(), label_s.cpu())
                if not np.isnan(mean_p): p_residuals.append(mean_p)
                if not np.isnan(mean_s): s_residuals.append(mean_s)

        val_loss   /= len(val_loader)
        mean_p_res  = float(np.mean(p_residuals)) if p_residuals else float("nan")
        mean_s_res  = float(np.mean(s_residuals)) if s_residuals else float("nan")

        print(
            f"Epoch {epoch:03d}/{n_epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"P_res={mean_p_res:.1f}samp | S_res={mean_s_res:.1f}samp | "
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
    """
    Run inference on a single (3, L) waveform (already normalized).

    Returns the full probability vectors, not just argmax — the curve shape
    encodes uncertainty and can be consumed as a pick PDF by location solvers.
    """
    model.eval()
    x = torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0).to(device)
    prob_p, prob_s = model(x)
    prob_p = prob_p[0].cpu().numpy()
    prob_s = prob_s[0].cpu().numpy()

    p_sample = int(prob_p.argmax())
    s_sample = int(prob_s.argmax())

    return {
        "prob_p":        prob_p,
        "prob_s":        prob_s,
        "p_sample":      p_sample,
        "s_sample":      s_sample,
        "p_confidence":  float(prob_p.max()),
        "s_confidence":  float(prob_s.max()),
        "sp_interval_s": (s_sample - p_sample) / 100.0,   # assumes 100 Hz
    }


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 64
    N_EPOCHS   = 50
    LR         = 1e-3

    # ── Dataloaders ───────────────────────────────────────────────────────
    # model_type="eqtransformer" sets window_len=6000 inside the pipeline.
    # This is NOT using the EQTransformer model — it just controls window length.
    train_pipe = SeisBenchPipelineWrapper(
        dataset_name="STEAD", split="train", model_type="eqtransformer",
        transformation_sigma=10, dataset_fraction=1.0,
    )
    val_pipe = SeisBenchPipelineWrapper(
        dataset_name="STEAD", split="dev", model_type="eqtransformer",
        transformation_sigma=10,
    )

    train_loader = train_pipe.get_dataloader(batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
    val_loader   = val_pipe.get_dataloader(batch_size=BATCH_SIZE,   num_workers=4, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────
    model = SeismicPicker(
        in_channels=3,
        base_channels=64,
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
        checkpoint_dir="checkpoints",
    )

    # ── Sanity check ──────────────────────────────────────────────────────
    dummy = np.random.randn(3, 6000).astype(np.float32)
    out   = predict_single(model, dummy, device=DEVICE)
    print(
        f"P: sample={out['p_sample']}  conf={out['p_confidence']:.3f} | "
        f"S: sample={out['s_sample']}  conf={out['s_confidence']:.3f} | "
        f"S-P={out['sp_interval_s']:.2f}s"
    )