import torch
import torch.nn as nn
import torch.nn.functional as F

class SeismicBiLSTM(nn.Module):
    """
    Pure BiLSTM seismic phase picker (no CNN frontend).

    Architecture:
        (B, 3, 6000)
        → Permute             : (B, 6000, 3)
        → 2-layer BiLSTM      : captures full temporal sequence from raw 3C inputs
        → Dropout
        → Two Linear heads    : independent P and S heads

    WARNING: Running LSTM directly on 6000 timesteps is computationally
    expensive and slow compared to a CNN downsampled approach.

    forward() returns raw **logits** (pre-sigmoid) so that BCEWithLogitsLoss can
    be applied directly. Call predict() explicitly when you need probabilities.
    """

    def __init__(
        self,
        in_channels: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        # BiLSTM: captures temporal patterns directly from raw signal
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_dropout = nn.Dropout(dropout)

        lstm_out = lstm_hidden * 2  # bidirectional doubles hidden size

        # Independent heads per phase
        self.head_p = nn.Linear(lstm_out, 1)
        self.head_s = nn.Linear(lstm_out, 1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, L) — normalized 3-component waveform from SeisBench

        Returns:
            logit_p: (B, L) — P-wave raw logits (pre-sigmoid)
            logit_s: (B, L) — S-wave raw logits (pre-sigmoid)
        """
        # x is (B, 3, L)
        x = x.permute(0, 2, 1)  # (B, L, 3) — LSTM expects (B, L, H)

        x, _ = self.lstm(x)     # (B, L, H)
        x = self.lstm_dropout(x)

        logit_p = self.head_p(x).squeeze(-1)  # (B, L)
        logit_s = self.head_s(x).squeeze(-1)  # (B, L)

        return logit_p, logit_s  # raw logits

    def predict(self, x: torch.Tensor):
        """
        Convenience wrapper: returns sigmoid probabilities instead of logits.
        """
        logit_p, logit_s = self.forward(x)
        return torch.sigmoid(logit_p), torch.sigmoid(logit_s)
