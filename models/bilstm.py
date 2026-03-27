import torch
import torch.nn as nn
import torch.nn.functional as F


class SeismicBiLSTM(nn.Module):
    """Pure BiLSTM seismic phase picker (no CNN frontend). Returns logits for P and S."""

    def __init__(
        self,
        in_channels: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_dropout = nn.Dropout(dropout)

        lstm_out = lstm_hidden * 2

        self.head_p = nn.Linear(lstm_out, 1)
        self.head_s = nn.Linear(lstm_out, 1)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.lstm_dropout(x)
        logit_p = self.head_p(x).squeeze(-1)
        logit_s = self.head_s(x).squeeze(-1)
        return logit_p, logit_s

    def predict(self, x: torch.Tensor):
        """Returns sigmoid probabilities. Equivalent to sigmoid(forward(x))."""
        logit_p, logit_s = self.forward(x)
        return torch.sigmoid(logit_p), torch.sigmoid(logit_s)
