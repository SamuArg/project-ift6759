import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
from tqdm import tqdm

try:
    from dataset.load_dataset import SeisBenchPipelineWrapper
except ImportError:
    # Fallback when running the file directly from project root
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dataset.load_dataset import SeisBenchPipelineWrapper


class ResidualConvBlock(nn.Module):
    """Two Conv1d layers with a skip connection and BatchNorm."""

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
    """CNN + BiLSTM seismic phase picker. Returns logit vectors (pre-sigmoid) for P and S."""

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

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
        )

        self.encoder = nn.Sequential(
            ResidualConvBlock(base_channels, kernel_size=7, dilation=1),
            ResidualConvBlock(base_channels, kernel_size=7, dilation=2),
            ResidualConvBlock(base_channels, kernel_size=7, dilation=4),
        )

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

        self.lstm = nn.LSTM(
            input_size=base_channels * 2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_dropout = nn.Dropout(dropout)

        lstm_out = lstm_hidden * 2
        head_in = lstm_out + 2 if self.use_coords else lstm_out

        self.head_p = nn.Conv1d(head_in, 1, kernel_size=1)
        self.head_s = nn.Conv1d(head_in, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, coords: torch.Tensor = None):
        B, C, L = x.shape

        x = self.stem(x)
        x = self.encoder(x)
        x = self.downsample(x)

        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.lstm_dropout(x)
        x = x.permute(0, 2, 1)

        if self.use_coords:
            if coords is None:
                raise ValueError(
                    "base_lstm instantiated with use_coords=True but no coords were provided to forward."
                )
            coords_expanded = coords.unsqueeze(2).expand(-1, -1, x.shape[2])
            x = torch.cat([x, coords_expanded], dim=1)

        logit_p = self.head_p(x).squeeze(1)
        logit_s = self.head_s(x).squeeze(1)

        logit_p = F.interpolate(
            logit_p.unsqueeze(1), size=L, mode="linear", align_corners=False
        ).squeeze(1)
        logit_s = F.interpolate(
            logit_s.unsqueeze(1), size=L, mode="linear", align_corners=False
        ).squeeze(1)

        return logit_p, logit_s

    def predict(self, x: torch.Tensor, coords: torch.Tensor = None):
        """Returns sigmoid probabilities. Equivalent to sigmoid(forward(x))."""
        logit_p, logit_s = self.forward(x, coords=coords)
        return torch.sigmoid(logit_p), torch.sigmoid(logit_s)
