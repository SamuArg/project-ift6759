import torch
import torch.nn as nn
import torch.nn.functional as F

class MagnitudePredictor(nn.Module):
    """
    CNN-BiLSTM for predicting earthquake magnitude from a 2-second (200-sample) waveform window.
    Architecture:
      Deep Conv1d blocks -> BiLSTM -> Global Average Pooling -> Deep Linear Regressor
    """
    def __init__(self, in_channels=3, base_channels=64, lstm_hidden=128, dropout=0.3):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Block 1: 200 -> 100
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 2: 100 -> 50
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(),
            nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 3: 50 -> 25
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(),
            nn.Conv1d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.lstm = nn.LSTM(
            input_size=base_channels * 4,
            hidden_size=lstm_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 1) # Predicts a single magnitude scalar
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, 3, 200) normalized waveform centered on P-wave
        Returns:
            mag: (B, 1) predicted scalar magnitude
        """
        x = self.encoder(x)             # (B, 256, 25)
        
        x = x.permute(0, 2, 1)          # (B, 25, 256) for LSTM
        x, _ = self.lstm(x)             # (B, 25, 256)
        
        x = x.mean(dim=1)               # Global Average Pooling over time (B, 256)
        
        mag = self.regressor(x)         # (B, 1)
        return mag

if __name__ == "__main__":
    # Test model shape
    model = MagnitudePredictor()
    x = torch.randn(16, 3, 200)
    out = model(x)
    print("Output shape:", out.shape) # Expected: (16, 1)
