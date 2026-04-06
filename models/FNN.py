import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
from tqdm import tqdm
from scipy.signal import stft
from scipy.signal import decimate
from pywt import cwt

from cwt_unet import CWTUNetPhasePicker

try:
    from dataset.load_dataset import SeisBenchPipelineWrapper
except ImportError:
    # Fallback when running the file directly from project root
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dataset.load_dataset import SeisBenchPipelineWrapper


def cwt_transform(traces, sf, down_sample_factor, cwt_widths=None):
    """
    Converts 3-component seismic traces to 3-channel time-frequency representations.
    
    Args:
        traces: numpy array of shape (3, num_samples)
        sf: sampling frequency in Hz
        cwt_widths: Array of wavelet scales to use
        
    Returns:
        amplitude_spectrogram: numpy array of shape (3, freq_bins, time_steps)
        log_spectrogram: log-scaled amplitude spectrogram
        (frequencies, times): Tuples of frequency and time axes
    """

    traces = decimate(traces, q=down_sample_factor, axis=-1)
    sf = sf / down_sample_factor
    num_channels, num_samples = traces.shape
        
    # Default widths (scales) if none are provided. 
    # In CWT, scale is inversely proportional to frequency.
    if cwt_widths is None:
        cwt_widths = np.arange(1, 129) 
        
    cwt_channels = []
    frequencies = None
    
    # 'mexh' is the PyWavelets identifier for the Mexican Hat (Ricker) wavelet
    wavelet = 'cmor1.5-1.0'
    
    for i in range(num_channels):
        # cwt calculates both the coefficients and the proper frequencies.
        # Passing sampling_period=1/sf ensures frequencies are returned in Hz.
        cwt_mat, freqs = cwt(traces[i], scales=cwt_widths, wavelet=wavelet, sampling_period=1/sf)
        cwt_channels.append(cwt_mat)
        
        # We only need to grab the frequency array once since it's identical for all 3 channels
        if frequencies is None:
            frequencies = freqs
            
    # Stack back into shape: (3, len(cwt_widths), num_samples)
    amplitude_spectrogram = np.abs(np.stack(cwt_channels))
    
    # For CWT, the time resolution perfectly matches the input trace length
    times = np.arange(num_samples) / sf

    # Log scaling to compress the dynamic range
    log_spectrogram = np.log1p(amplitude_spectrogram) 
    
    return amplitude_spectrogram, log_spectrogram, (frequencies, times)

def fourier_transform(traces, sf, nperseg=256):
    """
    Converts 3-component seismic traces to 3-channel time-frequency representations.
    
    Args:
        traces: numpy array of shape (3, num_samples)
        sf: sampling frequency in Hz
        nperseg: STFT window size
        
    Returns:
        amplitude_spectrogram: numpy array of shape (3, freq_bins, time_steps)
        log_spectrogram: log-scaled amplitude spectrogram
        (frequencies, times): Tuples of frequency and time axes
    """
    frequencies, times, Zxx = stft(traces, fs=sf, nperseg=nperseg)
    amplitude_spectrogram = np.abs(Zxx)
    log_spectrogram = np.log1p(amplitude_spectrogram) 
    
    return amplitude_spectrogram, log_spectrogram, (frequencies, times)

class SeismicPicker_fourier(nn.Module):
    """CNN seismic phase picker for fourier transform output. Returns logit vectors (pre-sigmoid) for P and S."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 16,
        num_layers: int = 4,
        use_coords: bool = False,
        initial_freq_bins: int = 129 # Default for stft with nperseg=256
    ):
        super().__init__()
        self.use_coords = use_coords

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        self.convs = nn.ModuleList()
        current_channels = base_channels
        remaining_freqs = initial_freq_bins
        remaining_freqs = remaining_freqs // 2
        for _ in range(num_layers):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=current_channels, out_channels=current_channels*2, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(current_channels*2),
                    nn.ReLU(inplace=True)
                )
                )
            current_channels = current_channels*2
            remaining_freqs = remaining_freqs // 2
        
        self.pool = nn.MaxPool2d(kernel_size=(2,1))
        self.conv_final = nn.Conv2d(in_channels=current_channels, out_channels=current_channels, kernel_size=(remaining_freqs, 1))
        
        cnn_out = current_channels
        head_in = cnn_out + 2 if self.use_coords else cnn_out
        self.head_p = nn.Conv1d(head_in, 1, kernel_size=1)
        self.head_s = nn.Conv1d(head_in, 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor, coords: torch.Tensor = None):
        """Return the logits of P and S waves for each time frame"""

        # x shape: (B, 3, F, T)
        B, C, FREQS, T = x.shape
        x = self.conv1(x)
        x = self.pool(x)

        # apply all the hidden layers
        for i, layer in enumerate(self.convs):
            x = F.relu(layer(x))
            x = self.pool(x)
            print(f"computed layer {i}")
        
        # Output shape should be: (B, 3, 1, T) -> Squeeze to (B, 3, T)
        x = self.conv_final(x)
        x = x.squeeze(2) 
        if self.use_coords:
            if coords is None:
                raise ValueError(
                    "fourien CNN instantiated with use_coords=True but no coords were provided."
                )
            coords_expanded = coords.unsqueeze(2).expand(-1, -1, x.shape[2])
            x = torch.cat([x, coords_expanded], dim=1)

        logit_p = self.head_p(x).squeeze(1)
        logit_s = self.head_s(x).squeeze(1)

        logit_p = F.interpolate(
            logit_p.unsqueeze(1), size=T, mode="linear", align_corners=False
        ).squeeze(1)
        logit_s = F.interpolate(
            logit_s.unsqueeze(1), size=T, mode="linear", align_corners=False
        ).squeeze(1)
        
        return logit_p, logit_s

    def predict(self, x: torch.Tensor, coords: torch.Tensor = None):
        """Returns sigmoid probabilities. Equivalent to sigmoid(forward(x))."""
        logit_p, logit_s = self.forward(x, coords=coords)
        return torch.sigmoid(logit_p), torch.sigmoid(logit_s)

modelCNN = SeismicPicker_fourier(
    in_channels=3,
    base_channels=16,
    num_layers=3,
    use_coords=False,
    initial_freq_bins=129
)
modelUnetSimple = CWTUNetPhasePicker(
    in_channels=3, 
    base_channels=16, 
    use_coords=False, 
    coord_channels=3, 
    simple=True
    )
modelUnetDouble = CWTUNetPhasePicker(
    in_channels=3, 
    base_channels=16, 
    use_coords=False, 
    coord_channels=3, 
    simple=False
    )
total_params = sum(p.numel() for p in modelUnetDouble.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")
dummy_spectrogram = torch.randn(1, 3, 129, 60) # Batch=1, Channels=3, Freq=129, Time=120
p_out, s_out = modelUnetDouble(dummy_spectrogram)
print(f"Output shape (P-wave logits): {p_out.shape}")
