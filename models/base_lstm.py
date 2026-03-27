"""
Seismic Phase Picker: CNN + BiLSTM for P-wave and S-wave arrival detection
Designed for STEAD and INSTANCE datasets.

Architecture overview:
    Raw 3C waveform → Conv1D encoder → BiLSTM → dual linear heads (P + S)

Output: two **logit** vectors of shape (batch, seq_len) — one per phase.
        Call .sigmoid() or use SeismicPicker.predict() to get probabilities.
        Returning logits lets the loss use BCEWithLogitsLoss directly, avoiding
        the numerically suboptimal sigmoid → clamp → logit round-trip.
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
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=pad, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=pad, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

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

    forward() returns raw **logits** (pre-sigmoid) so that BCEWithLogitsLoss can
    be applied directly without a sigmoid→logit round-trip.  Call predict() or
    apply .sigmoid() explicitly when you need probabilities.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
        use_coords: bool = False,
    ):
        super().__init__()
        self.use_coords = use_coords

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
            nn.Conv1d(
                base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(),
            nn.Conv1d(
                base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1
            ),
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

        head_in = lstm_out + 2 if self.use_coords else lstm_out

        # Independent heads per phase
        # WHY 1x1 conv (= linear per timestep): the LSTM already has full
        # temporal context; no wide kernel needed at this stage
        self.head_p = nn.Conv1d(head_in, 1, kernel_size=1)
        self.head_s = nn.Conv1d(head_in, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, coords: torch.Tensor = None):
        """
        Args:
            x: (B, 3, 6000) — normalized 3-component waveform from SeisBench

        Returns:
            logit_p: (B, 6000) — P-wave raw logits (pre-sigmoid)
            logit_s: (B, 6000) — S-wave raw logits (pre-sigmoid)

        Use predict() or .sigmoid() to obtain probabilities.
        """
        B, C, L = x.shape

        x = self.stem(x)  # (B, 64,  6000)
        x = self.encoder(x)  # (B, 64,  6000)
        x = self.downsample(x)  # (B, 128, 1500)

        x = x.permute(0, 2, 1)  # (B, 1500, 128) — LSTM expects (B, L, H)
        x, _ = self.lstm(x)  # (B, 1500, 256)
        x = self.lstm_dropout(x)
        x = x.permute(0, 2, 1)  # (B, 256, 1500) — back to (B, H, L) for Conv1d

        if self.use_coords:
            if coords is None:
                raise ValueError("base_lstm instantiated with use_coords=True but no coords were provided to forward.")
            # Map shape (B, 2) coords → (B, 2, 1500)
            coords_expanded = coords.unsqueeze(2).expand(-1, -1, x.shape[2])
            x = torch.cat([x, coords_expanded], dim=1) # Output (B, 258, 1500)

        logit_p = self.head_p(x).squeeze(1)  # (B, 1500)
        logit_s = self.head_s(x).squeeze(1)  # (B, 1500)

        # Upsample to original length.
        # WHY interpolation not transposed conv: smooth curves, no checkerboard
        # artifacts, no extra parameters needed. Upsample *logits* before sigmoid
        # so the sigmoid is only applied once at inference / in the loss.
        logit_p = F.interpolate(
            logit_p.unsqueeze(1), size=L, mode="linear", align_corners=False
        ).squeeze(
            1
        )  # (B, 6000)
        logit_s = F.interpolate(
            logit_s.unsqueeze(1), size=L, mode="linear", align_corners=False
        ).squeeze(
            1
        )  # (B, 6000)

        return logit_p, logit_s  # raw logits — each (B, 6000)

    def predict(self, x: torch.Tensor, coords: torch.Tensor = None):
        """
        Convenience wrapper: returns sigmoid probabilities instead of logits.
        Equivalent to calling sigmoid(forward(x)).

        Returns:
            prob_p: (B, 6000) — P-wave arrival probability
            prob_s: (B, 6000) — S-wave arrival probability
        """
        logit_p, logit_s = self.forward(x, coords=coords)
        return torch.sigmoid(logit_p), torch.sigmoid(logit_s)