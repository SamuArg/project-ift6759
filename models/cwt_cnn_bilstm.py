import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from cwt_unet import CWTBlock, Resample

class ResidualConv2dBlock(nn.Module):
    """2D Residual block for Time-Frequency data."""
    def __init__(self, channels: int, kernel_size=(3, 3), dilation=(1, 1)):
        super().__init__()
        # Calculate padding to keep output size same as input
        padding = (
            (kernel_size[0] - 1) // 2 * dilation[0],
            (kernel_size[1] - 1) // 2 * dilation[1]
        )
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)
    
class SeismicPicker2D(nn.Module):
    """2D CNN + BiLSTM seismic phase picker operating on CWT Spectrograms."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
        use_coords: bool = False,
        use_vs30: bool = False,
        use_instrument: bool = False,
    ):
        super().__init__()
        self.use_coords = use_coords
        self.use_vs30 = use_vs30
        self.use_instrument = use_instrument

        # 1. 2D STEM: Keep Time and Freq resolution the same initially
        # Kernel (3, 7) looks at a small frequency band but a wider time window
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
        )

        # 2. 2D ENCODER: Compress Frequency, Maintain Time
        # Stride (2, 1) halves the frequency dimension but keeps time untouched
        self.encoder = nn.Sequential(
            # Freq dim halves (e.g., 32 -> 16)
            nn.Conv2d(base_channels, base_channels, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            ResidualConv2dBlock(base_channels, dilation=(1, 1)),

            # Freq dim halves again (e.g., 16 -> 8)
            nn.Conv2d(base_channels, base_channels, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            ResidualConv2dBlock(base_channels, dilation=(1, 2)), # Dilate time axis slightly
        )

        # 3. 2D DOWNSAMPLE: Compress both Frequency and Time
        # Stride (2, 2) halves both dimensions, exactly like your old 1D downsample did for time
        self.downsample = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
        )

        # 4. FREQUENCY POOLING: Crush the last remaining frequency dimension to 1
        # E.g., shape goes from (B, 128, 2, T/4) to (B, 128, 1, T/4)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        # --- The rest remains identical to your original model ---
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
        head_in = lstm_out
        
        # Meta-data handling
        N_INSTRUMENT_CLASSES = 6
        if self.use_coords: head_in += 2                    
        if self.use_vs30: head_in += 1                    
        if self.use_instrument: head_in += N_INSTRUMENT_CLASSES 

        self.head_p = nn.Conv1d(head_in, 1, kernel_size=1)
        self.head_s = nn.Conv1d(head_in, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, coords: torch.Tensor = None, vs30: torch.Tensor = None, instrument: torch.Tensor = None):
        # Input shape from CWT: (B, 3, Freq_bins, L)
        B, C, F, L = x.shape

        # CNN Feature Extraction
        x = self.stem(x)           # -> (B, 64, Freq_bins, L)
        x = self.encoder(x)        # -> (B, 64, Freq_bins/4, L)
        x = self.downsample(x)     # -> (B, 128, Freq_bins/16, L/4)

        # Collapse the Frequency dimension 
        x = self.freq_pool(x)      # -> (B, 128, 1, L/4)
        
        # Squeeze out the 1-dimension to make it a 1D sequence for the LSTM
        x = x.squeeze(2)           # -> (B, 128, L/4)

        # Prepare for BiLSTM: (B, Time, Features)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.lstm_dropout(x)
        
        # Back to (B, Features, Time) for the 1D heads
        x = x.permute(0, 2, 1)

        T = x.shape[2]  
        features = [x]

        # Metadata Injection
        if self.use_coords:
            features.append(coords.unsqueeze(2).expand(-1, -1, T))
        if self.use_vs30:
            features.append(vs30.unsqueeze(2).expand(-1, -1, T))
        if self.use_instrument:
            features.append(instrument.unsqueeze(2).expand(-1, -1, T))

        if len(features) > 1:
            x = torch.cat(features, dim=1)  

        logit_p = self.head_p(x).squeeze(1)  
        logit_s = self.head_s(x).squeeze(1)  

        # Re-interpolate to the original sequence length L 
        logit_p = F.interpolate(
            logit_p.unsqueeze(1), size=L, mode="linear", align_corners=False
        ).squeeze(1)
        logit_s = F.interpolate(
            logit_s.unsqueeze(1), size=L, mode="linear", align_corners=False
        ).squeeze(1)

        return logit_p, logit_s

    def predict(self, x: torch.Tensor, coords: torch.Tensor = None, vs30: torch.Tensor = None, instrument: torch.Tensor = None):
        logit_p, logit_s = self.forward(
            x, coords=coords, vs30=vs30, instrument=instrument
        )
        return torch.sigmoid(logit_p), torch.sigmoid(logit_s)

class cwt_cnn_bilstm(nn.Module):
    """CNN + BiLSTM seismic phase picker that applies the continuous wavelet transform before the model. Returns logit vectors (pre-sigmoid) for P and S."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
        use_coords: bool = False,
        use_vs30: bool = False,
        use_instrument: bool = False,
    ):
        super().__init__()
        self.use_coords = use_coords
        self.use_vs30 = use_vs30
        self.use_instrument = use_instrument

        if torchaudio is not None:
            try:
                self.resampler = torchaudio.transforms.Resample(orig_freq=100, new_freq=50)
            except (RuntimeError, OSError):
                    self.resampler = Resample(orig_freq=100.0, new_freq=50.0)
        else:
            self.resampler = Resample(orig_freq=100.0, new_freq=50.0)
        self.cwtBlock = CWTBlock()

        self.base_lstm = SeismicPicker2D(
                in_channels=in_channels,
                base_channels=base_channels,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                dropout=dropout,
                use_coords=use_coords,
                use_vs30=use_vs30,
                use_instrument=use_instrument
            )
    
    def forward(
        self, 
        x: torch.Tensor, 
        coords: torch.Tensor = None, 
        vs30: torch.Tensor = None, 
        instrument: torch.Tensor = None
    ):
        # x shape: (B, 3, T)
        original_len = x.shape[2]
        
        # --- CWT pass ---
        x = self.resampler(x)
        x = self.cwtBlock(x)
        x = torch.log1p(x) # Log scaling for magnitude
        
        # --- CNN-BiLSTM Picker pass ---
        # x is now shape (B, 3, 32, T/2) natively ready for SeismicPicker2D
        logit_p, logit_s = self.base_lstm(
            x, 
            coords=coords, 
            vs30=vs30, 
            instrument=instrument
        )
        
        # --- Upsample pass ---
        # The picker outputs at the downsampled 50Hz temporal length.
        # We interpolate back to the 100Hz original_len.
        logit_p = F.interpolate(
            logit_p.unsqueeze(1), size=original_len, mode="linear", align_corners=False
        ).squeeze(1)
        
        logit_s = F.interpolate(
            logit_s.unsqueeze(1), size=original_len, mode="linear", align_corners=False
        ).squeeze(1)
        
        return logit_p, logit_s

    def predict(
        self, 
        x: torch.Tensor, 
        coords: torch.Tensor = None, 
        vs30: torch.Tensor = None, 
        instrument: torch.Tensor = None
    ):
        """Returns sigmoid probabilities at the original sampling rate."""
        logit_p, logit_s = self.forward(
            x, coords=coords, vs30=vs30, instrument=instrument
        )
        return torch.sigmoid(logit_p), torch.sigmoid(logit_s)



