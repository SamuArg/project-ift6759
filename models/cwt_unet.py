import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    import torchaudio
except (ImportError, RuntimeError, OSError):
    torchaudio = None


class Resample(nn.Module):
    def __init__(self, orig_freq, new_freq):
        super().__init__()
        self.scale_factor = new_freq / orig_freq

    def forward(self, x):
        # x shape: (B, C, T)
        return F.interpolate(
            x, scale_factor=self.scale_factor, mode="linear", align_corners=False
        )


class CWTBlock(nn.Module):
    def __init__(self, sf=50.0, num_bins=32, f_min=1.0, f_max=45.0, B=1.5, C=1.0):
        super().__init__()
        self.sf = sf

        # 1. Generate logarithmic frequency bins and corresponding scales
        freqs = torch.logspace(math.log10(f_min), math.log10(f_max), num_bins)
        scales = C / (freqs * (1.0 / sf))

        # 2. Define the temporal grid for the wavelet
        # The window length needs to be wide enough to capture the lowest frequency
        max_scale = scales[0]
        window_len = int(10 * max_scale)
        if window_len % 2 == 0:
            window_len += 1  # Ensure there is a perfect center point

        t = torch.arange(-window_len // 2 + 1, window_len // 2 + 1)

        real_kernels = []
        imag_kernels = []

        # 3. Construct the Complex Morlet wavelet mathematically for each scale
        for s in scales:
            x = t / s
            norm = 1.0 / math.sqrt(math.pi * B)
            gauss = torch.exp(-(x**2) / B)

            # Real and Imaginary components
            real_w = norm * torch.cos(2 * math.pi * C * x) * gauss
            imag_w = norm * torch.sin(2 * math.pi * C * x) * gauss

            # Normalize by 1/sqrt(s) to maintain energy across scales
            real_w = real_w / math.sqrt(s)
            imag_w = imag_w / math.sqrt(s)

            real_kernels.append(real_w)
            imag_kernels.append(imag_w)

        # 4. Store as PyTorch buffers so they move to the GPU automatically
        # Shape for Conv1d: (Out_channels, In_channels, Kernel_size) -> (32, 1, window_len)
        self.register_buffer(
            "real_kernels", torch.stack(real_kernels).unsqueeze(1).float()
        )
        self.register_buffer(
            "imag_kernels", torch.stack(imag_kernels).unsqueeze(1).float()
        )

    def forward(self, x):
        # Input shape: (Batch, 3 channels, Time)
        B, C, T = x.shape

        # Reshape to treat the 3 seismic channels as independent batch items
        # Shape becomes: (Batch * 3, 1, Time)
        x = x.view(B * C, 1, T)

        # 5. Apply the wavelet filters using 1D convolution
        # padding="same" ensures the temporal length remains unchanged
        real_out = F.conv1d(x, self.real_kernels, padding="same")
        imag_out = F.conv1d(x, self.imag_kernels, padding="same")

        # 6. Calculate the absolute magnitude of the complex result
        mag = torch.sqrt(real_out**2 + imag_out**2)

        # 7. Reshape back to the 2D Spectrogram format
        # Shape becomes: (Batch, 3 channels, 32 frequencies, Time)
        mag = mag.view(B, C, -1, T)

        return mag


class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class SimpleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.simple_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.simple_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, simple, max_kernel):
        super().__init__()
        if simple:
            conv_layer = SimpleConv(in_channels=in_channels, out_channels=out_channels)
        else:
            conv_layer = DoubleConv(in_channels, out_channels)
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(max_kernel), conv_layer)

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv with skip connections"""

    def __init__(
        self, in_channels, out_channels, simple, up_kernel=(2, 2), up_stride=(2, 2)
    ):
        super().__init__()
        # Use ConvTranspose2d to physically upsample the grid
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=up_kernel, stride=up_stride
        )
        if simple:
            self.conv = SimpleConv(in_channels, out_channels)
        else:
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Padding in case the input dimensions are not perfectly divisible by 2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate the skip connection (x2) along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CWTUNetPhasePicker(nn.Module):
    """
    Lightweight 2D U-Net for CWT spectrograms.
    Returns P and S wave logits at the original temporal resolution.
    """

    def __init__(
        self,
        in_channels=3,
        base_channels=16,
        use_coords=False,
        coord_channels=2,
        simple=False,
        cwt_onTheFly=True,
    ):
        super().__init__()
        self.use_coords = use_coords

        # --- CWT block ---
        self.cwt_onTheFly = cwt_onTheFly
        if cwt_onTheFly:
            if torchaudio is not None:
                try:
                    self.resampler = torchaudio.transforms.Resample(
                        orig_freq=100, new_freq=50
                    )
                except (RuntimeError, OSError):
                    self.resampler = Resample(orig_freq=100.0, new_freq=50.0)
            else:
                self.resampler = Resample(orig_freq=100.0, new_freq=50.0)
            self.cwtBlock = CWTBlock()

        # --- Encoder (Downsampling) ---
        if simple:
            self.inc = SimpleConv(in_channels, base_channels)
        else:
            self.inc = DoubleConv(in_channels, base_channels)  # 3 -> 16
        self.down1 = Down(base_channels, base_channels * 2, simple, (2, 1))  # 16 -> 32
        self.down2 = Down(base_channels * 2, base_channels * 4, simple, 2)  # 32 -> 64
        self.down3 = Down(base_channels * 4, base_channels * 8, simple, 2)  # 64 -> 128

        # --- Decoder (Upsampling) ---
        self.up1 = Up(base_channels * 8, base_channels * 4, simple)  # 128 -> 64
        self.up2 = Up(base_channels * 4, base_channels * 2, simple)  # 64 -> 32
        self.up3 = Up(
            base_channels * 2, base_channels, simple, up_kernel=(2, 1), up_stride=(2, 1)
        )  # 32 -> 16

        self.freq_attention = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=1),
            nn.Softmax(dim=2),  # Apply softmax across the frequency axis (dim=2)
        )

        # --- Output Heads ---
        # We calculate the head input channels.
        head_in_channels = base_channels + (coord_channels if use_coords else 0)

        self.head_p = nn.Conv1d(head_in_channels, 1, kernel_size=1)
        self.head_s = nn.Conv1d(head_in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, coords: torch.Tensor = None):
        # x shape: (B, 3, T)
        original_len = x.shape[2]

        # --- CWT pass ---
        if self.cwt_onTheFly:
            x = self.resampler(x)
            x = self.cwtBlock(x)
            x = torch.log1p(x)
        # --- Encoder pass ---
        x1 = self.inc(x)  # Skip 1
        x2 = self.down1(x1)  # Skip 2
        x3 = self.down2(x2)  # Skip 3
        x4 = self.down3(x3)  # Bottleneck

        # --- Decoder pass ---
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)  # Output shape: (B, 16, F, T)

        ## --- Frequency Compression via Soft Attention ---
        # 1. Compute attention weights: shape (B, 1, F, T)
        attn_weights = self.freq_attention(x)

        # 2. Multiply features by weights, then sum over the frequency dimension
        # x * attn_weights broadcasts the weights across the 16 channels.
        # .sum(dim=2) collapses the frequency axis.
        # Output shape: (B, 16, T)
        x = (x * attn_weights).sum(dim=2)

        # --- Coordinate Integration ---
        if self.use_coords:
            if coords is None:
                raise ValueError(
                    "U-Net instantiated with use_coords=True but no coords provided."
                )
            coords_expanded = coords.unsqueeze(2).expand(-1, -1, x.shape[2])
            x = torch.cat([x, coords_expanded], dim=1)

        # --- Final Logits ---
        # Shape goes from (B, C, T) -> (B, 1, T) -> (B, T)
        logit_p = self.head_p(x).squeeze(1)
        logit_s = self.head_s(x).squeeze(1)

        # --- Rescale to original length if needed ---
        if logit_p.shape[1] != original_len:
            logit_p = F.interpolate(
                logit_p.unsqueeze(1),
                size=original_len,
                mode="linear",
                align_corners=False,
            ).squeeze(1)
            logit_s = F.interpolate(
                logit_s.unsqueeze(1),
                size=original_len,
                mode="linear",
                align_corners=False,
            ).squeeze(1)

        return logit_p, logit_s

    def predict(self, x: torch.Tensor, coords: torch.Tensor = None):
        logit_p, logit_s = self.forward(x, coords=coords)
        return torch.sigmoid(logit_p), torch.sigmoid(logit_s)
