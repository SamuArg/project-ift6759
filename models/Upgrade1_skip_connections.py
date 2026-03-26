import torch
import torch.nn as nn
from models.unet_parts import EncoderBlock, DecoderBlock

class SeismicPickerUNet(nn.Module):
    """
    U-Net CNN + BiLSTM bottleneck seismic phase picker. (From Step 1 Upgrade)

    Channel progression (base_ch=32):
        Encoder:  3 -> 32 -> 64 -> 128 -> 256  (4 stages, each halves seq length)
        Bottleneck: BiLSTM at (B, 256, 750) -> processes (B, 750, 256) -> back to (B, 256, 750)
        Decoder:  256+128->128 -> 128+64->64 -> 64+32->32 -> upsample to 6000
        Heads: two independent 1x1 convs -> sigmoid
    """
    def __init__(
        self,
        in_channels: int   = 3,
        base_ch: int       = 32,
        lstm_hidden: int   = 128,
        lstm_layers: int   = 2,
        dropout: float     = 0.2,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(),
        )

        self.enc1 = EncoderBlock(base_ch,     base_ch * 2,  dilation=1)
        self.enc2 = EncoderBlock(base_ch * 2, base_ch * 4,  dilation=2)
        self.enc3 = EncoderBlock(base_ch * 4, base_ch * 8,  dilation=4)

        bottleneck_ch = base_ch * 8

        self.lstm = nn.LSTM(
            input_size=bottleneck_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_dropout = nn.Dropout(dropout)
        self.lstm_proj = nn.Sequential(
            nn.Conv1d(lstm_hidden * 2, bottleneck_ch, kernel_size=1),
            nn.BatchNorm1d(bottleneck_ch),
            nn.ReLU(),
        )

        self.dec1 = DecoderBlock(bottleneck_ch, base_ch * 4, base_ch * 4)
        self.dec2 = DecoderBlock(base_ch * 4,  base_ch * 2, base_ch * 2)
        self.dec3 = DecoderBlock(base_ch * 2,  base_ch,     base_ch)

        self.final_conv = nn.Sequential(
            nn.Conv1d(base_ch, base_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(),
        )

        self.head_p = nn.Conv1d(base_ch, 1, kernel_size=1)
        self.head_s = nn.Conv1d(base_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)

        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.lstm_dropout(x)
        x = x.permute(0, 2, 1)
        x = self.lstm_proj(x)

        x = self.dec1(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec3(x, skip1)

        x = self.final_conv(x)

        prob_p = torch.sigmoid(self.head_p(x).squeeze(1))
        prob_s = torch.sigmoid(self.head_s(x).squeeze(1))

        return prob_p, prob_s