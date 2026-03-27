import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet_parts import EncoderBlock, DecoderBlock

# CONSTANTS
SAMPLE_RATE = 100
MIN_SP_SAMPLES = 50
MAX_SP_SAMPLES = 4000
S_LABEL_SIGMA = 20


# INPUT FEATURE ENGINEERING
def build_s_input(waveform: torch.Tensor, p_prob: torch.Tensor) -> torch.Tensor:
    """
    Constructs the 5-channel input tensor for the S-wave model.
    Channels: Z, N, E, P-prob, Horizontal resultant
    """
    Z = waveform[:, 0:1, :]
    N = waveform[:, 1:2, :]
    E = waveform[:, 2:3, :]

    H = torch.sqrt(N**2 + E**2 + 1e-9)
    H_std = H.std(dim=-1, keepdim=True) + 1e-9
    H = H / H_std

    p_ch = p_prob.unsqueeze(1)
    return torch.cat([Z, N, E, p_ch, H], dim=1)


def build_sp_mask(
    p_prob: torch.Tensor,
    min_sp: int = MIN_SP_SAMPLES,
    max_sp: int = MAX_SP_SAMPLES,
    confidence_threshold: float = 0.1,
) -> torch.Tensor:
    """
    Physics-based mask that zeroes out S-probability predictions in
    physically implausible regions.
    """
    B, L = p_prob.shape
    device = p_prob.device

    t_indices = torch.arange(L, device=device).float()
    peak = p_prob.max(dim=-1, keepdim=True).values
    weights = torch.clamp(p_prob - 0.5 * peak, min=0.0)
    weight_sum = weights.sum(dim=-1, keepdim=True) + 1e-9
    t_P = (weights * t_indices).sum(dim=-1) / weight_sum.squeeze(1)
    t_P = t_P.long()

    mask = torch.zeros(B, L, device=device)
    t_start = (t_P + min_sp).clamp(0, L - 1)
    t_end = (t_P + max_sp).clamp(0, L - 1)

    for i in range(B):
        if peak[i, 0] > confidence_threshold:
            mask[i, t_start[i] : t_end[i]] = 1.0
        else:
            mask[i, :] = 1.0

    return mask


# S-MODEL
class CascadeSPicker(nn.Module):
    """
    Dedicated S-wave picker for the cascade pipeline.
    Expects 5-channel input from build_s_input.
    """

    def __init__(
        self, in_channels=5, base_ch=32, lstm_hidden=128, lstm_layers=2, dropout=0.2
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(),
        )
        self.enc1 = EncoderBlock(base_ch, base_ch * 2, dilation=1)
        self.enc2 = EncoderBlock(base_ch * 2, base_ch * 4, dilation=2)
        self.enc3 = EncoderBlock(base_ch * 4, base_ch * 8, dilation=4)
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
        self.dec2 = DecoderBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.dec3 = DecoderBlock(base_ch * 2, base_ch, base_ch)
        self.final_conv = nn.Sequential(
            nn.Conv1d(base_ch, base_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(),
        )
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

        prob_s = torch.sigmoid(self.head_s(x).squeeze(1))
        return prob_s


# LOSS
def s_wave_loss(
    prob_s: torch.Tensor,
    label_s: torch.Tensor,
    sp_mask: torch.Tensor,
    pos_weight_factor: float = 10.0,
    mask_weight: float = 0.1,
) -> torch.Tensor:
    """
    S-wave loss comprising picking loss, suppression loss, and SP-consistency loss.
    """
    pos_weight = torch.tensor(pos_weight_factor, device=prob_s.device)

    # 1. Picking loss
    inside = sp_mask
    pick_weight = (1 + (pos_weight - 1) * label_s) * inside
    loss_pick = F.binary_cross_entropy(
        prob_s, label_s, weight=pick_weight, reduction="sum"
    )
    loss_pick = loss_pick / inside.sum().clamp(min=1)

    # 2. Suppression loss
    outside = 1.0 - sp_mask
    target_outside = torch.zeros_like(prob_s)
    loss_suppress = F.binary_cross_entropy(
        prob_s, target_outside, weight=outside, reduction="sum"
    )
    loss_suppress = loss_suppress / outside.sum().clamp(min=1)

    # 3. SP-consistency
    loss_consistency = (prob_s * (1 - sp_mask)).mean()

    return loss_pick + mask_weight * loss_suppress + mask_weight * loss_consistency
