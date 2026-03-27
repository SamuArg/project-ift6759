import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    """
    Two Conv1d layers with BatchNorm and a residual skip.
    Used in both the encoder and decoder paths.
    """

    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=pad, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=pad, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        r = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + r)


class EncoderBlock(nn.Module):
    """
    One U-Net encoder stage:
        ResidualConvBlock (at current resolution) -> strided Conv1d (downsample ×2)
    Returns BOTH the pre-downsample features (the skip connection) AND the
    downsampled features (passed to the next stage).
    """

    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int = 7, dilation: int = 1
    ):
        super().__init__()
        self.res = ResidualConvBlock(in_ch, kernel_size=kernel_size, dilation=dilation)
        # Strided conv: learned downsampling, halves sequence length
        self.down = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.res(x)
        skip = x
        x = self.down(x)
        return x, skip


class DecoderBlock(nn.Module):
    """
    One U-Net decoder stage:
        Upsample ×2 -> concat skip -> Conv1d to merge -> ResidualConvBlock
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, kernel_size: int = 7):
        super().__init__()
        self.merge = nn.Sequential(
            nn.Conv1d(in_ch + skip_ch, out_ch, kernel_size=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )
        self.res = ResidualConvBlock(out_ch, kernel_size=kernel_size, dilation=1)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.merge(x)
        x = self.res(x)
        return x
