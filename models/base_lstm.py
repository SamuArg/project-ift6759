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
from tqdm import tqdm

# Import the canonical pipeline — do NOT redefine it here.
# Importing from dataset.load_dataset ensures we always use the same
# windowing, normalization, and SteeredGenerator logic for all models.
try:
    from dataset.load_dataset import SeisBenchPipelineWrapper
except ImportError:
    # Fallback when running the file directly from project root
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dataset.load_dataset import SeisBenchPipelineWrapper



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

    WHY binary_cross_entropy_with_logits instead of binary_cross_entropy:
    F.binary_cross_entropy is unsafe with AMP (autocast) because float16 can
    produce NaN/Inf inside the log. BCEWithLogitsLoss fuses sigmoid + log into a
    numerically stable log-sum-exp computation that is AMP-safe.
    We convert our sigmoid probabilities back to logits first; the result is
    mathematically identical to the original BCE but compatible with autocast.
    """
    pos_weight = torch.tensor(pos_weight_factor, device=pred.device)
    weight     = 1 + (pos_weight - 1) * target
    # Clamp avoids logit(-inf/+inf) for perfectly saturated sigmoid outputs
    logits = torch.logit(pred.clamp(min=1e-6, max=1 - 1e-6))
    return F.binary_cross_entropy_with_logits(logits, target, weight=weight)


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
# INFERENCE HELPER
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_single(model: nn.Module, waveform: np.ndarray, device=None) -> dict:
    """
    Run inference on a single (3, L) waveform (already normalized).

    Returns the full probability vectors, not just argmax — the curve shape
    encodes uncertainty and can be consumed as a pick PDF by location solvers.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device) if isinstance(device, str) else device

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
