import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.ReLU(inplace=True)
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
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.simple_conv(x)
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, simple):
        super().__init__()
        if simple:
            conv_layer = SimpleConv(in_channels=in_channels, out_channels=out_channels)
        else:
            conv_layer = DoubleConv(in_channels, out_channels)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_layer
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv with skip connections"""
    def __init__(self, in_channels, out_channels, simple):
        super().__init__()
        # Use ConvTranspose2d to physically upsample the grid
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        if simple:
            self.conv = SimpleConv(in_channels, out_channels)
        else:
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Padding in case the input dimensions are not perfectly divisible by 2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate the skip connection (x2) along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetPhasePicker(nn.Module):
    """
    Lightweight 2D U-Net for CWT spectrograms.
    Returns P and S wave logits at the original temporal resolution.
    """
    def __init__(self, in_channels=3, base_channels=16, use_coords=False, coord_channels=3, simple=False):
        super().__init__()
        self.use_coords = use_coords
        
        # --- Encoder (Downsampling) ---
        if simple:
            self.inc = SimpleConv(in_channels, base_channels)
        else:
            self.inc = DoubleConv(in_channels, base_channels)             # 3 -> 16
        self.down1 = Down(base_channels, base_channels * 2, simple)           # 16 -> 32
        self.down2 = Down(base_channels * 2, base_channels * 4, simple)       # 32 -> 64
        self.down3 = Down(base_channels * 4, base_channels * 8, simple)       # 64 -> 128
        
        # --- Decoder (Upsampling) ---
        self.up1 = Up(base_channels * 8, base_channels * 4, simple)           # 128 -> 64
        self.up2 = Up(base_channels * 4, base_channels * 2, simple)           # 64 -> 32
        self.up3 = Up(base_channels * 2, base_channels, simple)               # 32 -> 16
        
        # --- Output Heads ---
        # We calculate the head input channels. 
        head_in_channels = base_channels + (coord_channels if use_coords else 0)
        
        self.head_p = nn.Conv1d(head_in_channels, 1, kernel_size=1)
        self.head_s = nn.Conv1d(head_in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, coords: torch.Tensor = None):
        # x shape: (B, 3, F, T) -> e.g., (Batch, 3, 128, 12000)
        
        # --- Encoder pass ---
        x1 = self.inc(x)       # Skip 1
        x2 = self.down1(x1)    # Skip 2
        x3 = self.down2(x2)    # Skip 3
        x4 = self.down3(x3)    # Bottleneck
        
        # --- Decoder pass ---
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)    # Output shape: (B, 16, F, T)
        
        # --- Frequency Compression ---
        # Instead of a Conv2d, we take the max activation across the frequency 
        # dimension (dim=2). This removes the frequency axis entirely, leaving (B, 16, T).
        # This is brilliant because it works dynamically for any number of frequency bins!
        x = torch.max(x, dim=2)[0] 
        
        # --- Coordinate Integration ---
        if self.use_coords:
            if coords is None:
                raise ValueError("U-Net instantiated with use_coords=True but no coords provided.")
            coords_expanded = coords.unsqueeze(2).expand(-1, -1, x.shape[2])
            x = torch.cat([x, coords_expanded], dim=1)

        # --- Final Logits ---
        # Shape goes from (B, C, T) -> (B, 1, T) -> (B, T)
        logit_p = self.head_p(x).squeeze(1)
        logit_s = self.head_s(x).squeeze(1)
        
        return logit_p, logit_s

    def predict(self, x: torch.Tensor, coords: torch.Tensor = None):
        logit_p, logit_s = self.forward(x, coords=coords)
        return torch.sigmoid(logit_p), torch.sigmoid(logit_s)