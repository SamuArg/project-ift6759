"""
CASCADE S-WAVE PICKER
=====================
A dedicated S-wave model that uses the P-wave model's output as prior knowledge.

Core idea:
    The P-picker is already excellent. S-wave picking is hard because the model
    must find an onset against P-wave coda (dirty-signal vs dirtier-signal).
    By telling the S-model WHERE the P-wave is, we drastically reduce its
    search space and give it a physics-grounded anchor.

    Two-stage pipeline:
        Stage 1 (frozen): P-model → P-probability vector (B, 6000)
        Stage 2 (trained): S-model takes (B, 5, 6000) input:
                               ch 0-2 : original ZNE waveform
                               ch 3   : P-probability from Stage 1
                               ch 4   : horizontal resultant sqrt(N²+E²)
                           → S-probability vector (B, 6000)
                           → physics mask zeros everything before t_P + min_SP

What is NEW vs Step 1 U-Net:
    1. P-model is run first; its output is concatenated as an extra input channel
    2. Horizontal resultant channel added (S energy is on horizontals)
    3. Physics-based SP mask applied at inference AND during training
    4. S-specific label sigma (20 samples = 0.2s, wider than P's 10)
    5. SP-consistency loss term that penalises S picks before the P pick
    6. The P-model weights are FROZEN during S-model training

Architecture:
    Identical to Step 1 U-Net (ResBlock encoder → BiLSTM bottleneck → decoder)
    EXCEPT in_channels=5 instead of 3.
    That single change is the entire architectural difference. Everything else
    is the same — the U-Net skip connections, the BiLSTM bottleneck, the
    independent heads. The new signal lives entirely in the 2 extra input channels.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path

import seisbench.data as sbd
import seisbench.generate as sbg


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE    = 100       # Hz
SEQ_LEN        = 6000      # samples (60 s)

# Physics constraint: Vp/Vs ≈ 1.73 in average crust → S arrives ~0.73× later
# than P relative to source. For local events (< 200 km), the minimum S-P
# interval is rarely < 1 s. We use a conservative 0.5 s (50 samples) to avoid
# masking legitimate close-in events.
MIN_SP_SAMPLES = 50        # 0.5 s minimum S-P interval

# Maximum physically plausible S-P for events within a 60-s window.
# At crustal Vp=6 km/s, Vs=3.5 km/s, delta_v=2.5 km/s:
# max_distance ≈ 59 s × 2.5 km/s = ~150 km → S-P ≈ 40 s → 4000 samples.
# We leave the upper end open (the window edge acts as the natural limit).
MAX_SP_SAMPLES = 4000

# Label sigma for S-wave training.
# WHY 20 (0.2 s) vs 10 (0.1 s) for P:
#   Human S-picks in STEAD and INSTANCE have higher uncertainty than P-picks.
#   P-picks are anchored to the first sharp onset; S-picks require recognising
#   a change in an already-disturbed signal. Published inter-analyst S-pick
#   scatter is 0.1–0.3 s, so sigma=20 (0.2 s) better matches the label noise.
#   Training with too tight a sigma punishes the model for picks that are
#   actually within human uncertainty.
S_LABEL_SIGMA  = 20


# ──────────────────────────────────────────────────────────────────────────────
# DATA PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

class SeisBenchPipelineWrapper:
    """
    Standard SeisBench wrapper (unchanged from previous steps).
    Returns batches with keys: X, y_p, y_s.
    We use y_p as the ground-truth P label (for the frozen P-model's targets,
    useful for debugging) and y_s as the S-model's training target.
    """
    def __init__(self, dataset_name="STEAD", split="train",
                 model_type="eqtransformer", component_order="ZNE",
                 max_distance=None, transformation_shape="gaussian",
                 transformation_sigma=10, s_sigma=None, dataset_fraction=1.0):
        """
        s_sigma: if provided, uses a DIFFERENT sigma for the S label than
                 for the P label. This is the main addition vs previous steps.
                 WHY separate sigma per phase: P and S picks have different
                 uncertainty characteristics; forcing the same sigma is a
                 misspecification of the label distribution.
        """
        self.dataset_name         = dataset_name.upper()
        self.split                = split.lower()
        self.model_type           = model_type.lower()
        self.component_order      = component_order
        self.max_distance         = max_distance
        self.transformation_shape = transformation_shape
        self.transformation_sigma = transformation_sigma
        self.s_sigma              = s_sigma if s_sigma is not None else transformation_sigma
        self.dataset_fraction     = dataset_fraction

        if self.dataset_name == "STEAD":
            dataset = sbd.STEAD(component_order=self.component_order)
        elif self.dataset_name == "INSTANCE":
            dataset = sbd.InstanceCountsCombined(component_order=self.component_order)
        elif self.dataset_name == "VCSEIS":
            dataset = sbd.VCSEIS(component_order=self.component_order)
        elif self.dataset_name == "GEOFON":
            dataset = sbd.GEOFON(component_order=self.component_order)
        elif self.dataset_name == "TXED":
            dataset = sbd.TXED(component_order=self.component_order)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        if self.split == "train":
            self.dataset = dataset.train()
        elif self.split in ["dev", "val", "validation"]:
            self.dataset = dataset.dev()
        elif self.split == "test":
            self.dataset = dataset.test()

        if self.max_distance is not None and self.dataset_name in ["STEAD", "INSTANCE"]:
            possible_cols = ["path_hyp_distance_km", "source_distance_km"]
            dist_col = next((c for c in possible_cols if c in self.dataset.metadata.columns), None)
            if dist_col:
                mask = (
                    (self.dataset.metadata[dist_col] <= self.max_distance) |
                    (self.dataset.metadata[dist_col].isna())
                ).values
                self.dataset.filter(mask)

        if 0.0 < self.dataset_fraction < 1.0:
            total = len(self.dataset)
            n     = int(total * self.dataset_fraction)
            mask  = np.zeros(total, dtype=bool)
            mask[np.random.choice(total, n, replace=False)] = True
            self.dataset.filter(mask)

        self.generator = sbg.GenericGenerator(self.dataset)
        self._attach_pipeline()

    def _attach_pipeline(self):
        window_len = 6000 if self.model_type == "eqtransformer" else 3001

        p_col = "trace_p_arrival_sample"
        s_col = "trace_s_arrival_sample"
        if p_col not in self.dataset.metadata.columns:
            p_col = "trace_P_arrival_sample"
            s_col = "trace_S_arrival_sample"

        augmentations = [
            sbg.WindowAroundSample(
                metadata_keys=[p_col, s_col], selection="random",
                samples_before=window_len // 2, strategy="pad",
                windowlen=window_len,
            ),
            sbg.ChangeDtype(np.float32),
            sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="std"),
            # P label: tighter sigma (default 10 = 0.1s)
            sbg.ProbabilisticLabeller(
                label_columns=[p_col], shape=self.transformation_shape,
                sigma=self.transformation_sigma, key=("X", "y_p"), dim=0,
            ),
            sbg.ChangeDtype(np.float32, key="y_p"),
            # S label: wider sigma (default 20 = 0.2s)
            # WHY different sigma per phase: see S_LABEL_SIGMA comment above.
            sbg.ProbabilisticLabeller(
                label_columns=[s_col], shape=self.transformation_shape,
                sigma=self.s_sigma, key=("X", "y_s"), dim=0,
            ),
            sbg.ChangeDtype(np.float32, key="y_s"),
            sbg.ChangeDtype(np.float32, key="X"),
        ]
        self.generator.add_augmentations(augmentations)

    def get_dataloader(self, batch_size=32, num_workers=4, shuffle=True):
        return DataLoader(self.generator, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)


# ──────────────────────────────────────────────────────────────────────────────
# INPUT FEATURE ENGINEERING  (runs on GPU, inside the training loop)
# ──────────────────────────────────────────────────────────────────────────────

def build_s_input(waveform: torch.Tensor, p_prob: torch.Tensor) -> torch.Tensor:
    """
    Constructs the 5-channel input tensor for the S-wave model.

    Args:
        waveform: (B, 3, L) — normalized ZNE waveform (from SeisBench)
        p_prob:   (B, L)    — P-arrival probability from the frozen P-model

    Returns:
        x: (B, 5, L) — 5-channel input to the S-model

    Channel breakdown:
        ch 0: Z  component (vertical)
        ch 1: N  component (north)
        ch 2: E  component (east)
        ch 3: P-probability vector
        ch 4: Horizontal resultant = sqrt(N² + E²)

    WHY ch 3 — P-probability:
        The S-model needs to know where P arrived to apply the physics mask
        and to condition its search. Feeding the full probability vector (not
        just the argmax) is important because:
        - It carries uncertainty: a wide peak tells the S-model "P is
          somewhere in this range, be careful near the edges."
        - It is differentiable: if we ever want to fine-tune P and S jointly,
          the gradient can flow from the S-model back to the P-model.
        - It is scale-consistent: the P-model's sigmoid output is in [0,1],
          matching the waveform's normalized amplitude range.

    WHY ch 4 — horizontal resultant:
        S-waves are transverse — they displace the ground perpendicular to
        the propagation direction. For a horizontally propagating wave, this
        means maximum displacement is on the horizontal components (N, E).
        The vertical component primarily sees P-wave energy.
        sqrt(N² + E²) is the instantaneous horizontal amplitude envelope.
        It has a characteristic onset at the S-arrival that is often SHARPER
        than either individual horizontal component, because it is invariant
        to the horizontal azimuth of the station relative to the source.
        This means the S-onset appears as a step function in the horizontal
        resultant regardless of station geometry — a much cleaner target.

        WHY not just give it N and E separately (which the model already has):
        The model CAN learn sqrt(N²+E²) from N and E in principle, but:
        - It requires the model to learn a nonlinear combination (sqrt of sum
          of squares) which takes many training examples to approximate well.
        - By precomputing it, we give the model a direct physics-derived
          feature, accelerating convergence and improving low-SNR performance.
    """
    Z = waveform[:, 0:1, :]   # (B, 1, L)
    N = waveform[:, 1:2, :]   # (B, 1, L)
    E = waveform[:, 2:3, :]   # (B, 1, L)

    # Horizontal resultant: sqrt(N² + E²), normalized to unit std
    H = torch.sqrt(N ** 2 + E ** 2 + 1e-9)   # (B, 1, L)
    H_std = H.std(dim=-1, keepdim=True) + 1e-9
    H = H / H_std                              # normalize independently

    # P-probability as an extra channel: (B, L) → (B, 1, L)
    p_ch = p_prob.unsqueeze(1)                 # (B, 1, L)

    # Concatenate all 5 channels along dim=1
    return torch.cat([Z, N, E, p_ch, H], dim=1)   # (B, 5, L)


def build_sp_mask(p_prob: torch.Tensor,
                  min_sp: int = MIN_SP_SAMPLES,
                  max_sp: int = MAX_SP_SAMPLES,
                  confidence_threshold: float = 0.1) -> torch.Tensor:
    """
    Physics-based mask that zeroes out S-probability predictions in
    physically implausible regions.

    Concretely, for each trace in the batch:
        - If the P-model is confident (peak P-prob > threshold):
              mask = 0  for t < t_P + min_sp
              mask = 0  for t > t_P + max_sp
              mask = 1  for t in [t_P + min_sp, t_P + max_sp]
        - If the P-model is not confident (noise trace or no clear P):
              mask = 1 everywhere (don't constrain the S-model)

    WHY apply the mask to the LOGITS before sigmoid rather than after:
        Multiplying a sigmoid output by a binary mask produces 0 in masked
        regions, which BCE interprets as a confident prediction of "no S here."
        This contributes strongly to the gradient and can overwhelm the
        actual S-picking signal. Instead, we apply the mask as a multiplicative
        gate on the probability AFTER sigmoid, and handle masked regions
        separately in the loss (see s_wave_loss).

    WHY not hard-code t_P as argmax:
        The P-model's argmax can be off by a few samples due to noise.
        We find the CENTER OF MASS of the P-probability curve within the top
        percentile — this is more robust to multimodal probability distributions
        and gives a smoother estimate of t_P.

    Args:
        p_prob: (B, L) — P-arrival probability from frozen P-model
        min_sp: minimum S-P interval in samples
        max_sp: maximum S-P interval in samples
        confidence_threshold: minimum peak P-prob to apply the mask

    Returns:
        mask: (B, L) float tensor, values in {0, 1}
    """
    B, L = p_prob.shape
    device = p_prob.device

    # Estimate t_P per trace via center-of-mass of top-50% of P-prob mass
    # WHY center of mass not argmax: argmax is brittle to flat or bimodal
    # probability curves. CoM gives a smoother, more stable estimate.
    t_indices = torch.arange(L, device=device).float()   # (L,)

    # Threshold at 50% of peak to focus on the main peak region
    peak = p_prob.max(dim=-1, keepdim=True).values         # (B, 1)
    weights = torch.clamp(p_prob - 0.5 * peak, min=0.0)    # (B, L), zero below half-peak
    weight_sum = weights.sum(dim=-1, keepdim=True) + 1e-9   # (B, 1)
    t_P = (weights * t_indices).sum(dim=-1) / weight_sum.squeeze(1)  # (B,)
    t_P = t_P.long()                                        # (B,) integer sample indices

    # Build mask: 1 in allowed window, 0 outside
    mask = torch.zeros(B, L, device=device)
    t_start = (t_P + min_sp).clamp(0, L - 1)   # (B,)
    t_end   = (t_P + max_sp).clamp(0, L - 1)   # (B,)

    for i in range(B):
        if peak[i, 0] > confidence_threshold:
            mask[i, t_start[i]:t_end[i]] = 1.0
        else:
            # Low P-confidence → noise trace → no constraint
            mask[i, :] = 1.0

    return mask   # (B, L)


# ──────────────────────────────────────────────────────────────────────────────
# MODEL BUILDING BLOCKS  (identical to Step 1 — U-Net components)
# ──────────────────────────────────────────────────────────────────────────────

class ResidualConvBlock(nn.Module):
    """
    Two Conv1d layers with BatchNorm and a residual skip.
    No changes from Step 1 — the block is architecture-agnostic.
    """
    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(channels)
        self.bn2   = nn.BatchNorm1d(channels)

    def forward(self, x):
        r = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + r)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        self.res  = ResidualConvBlock(in_ch, kernel_size=kernel_size, dilation=dilation)
        self.down = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        x    = self.res(x)
        skip = x
        x    = self.down(x)
        return x, skip


class DecoderBlock(nn.Module):
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


# ──────────────────────────────────────────────────────────────────────────────
# P-MODEL  (Step 1 architecture, loaded frozen)
# ──────────────────────────────────────────────────────────────────────────────

class SeismicPickerUNet(nn.Module):
    """
    Identical to Step 1. Used as the frozen P-model in the cascade.
    in_channels=3 (ZNE only — no extra channels).

    WHY freeze the P-model during S-model training:
        If we fine-tune the P-model simultaneously, the S-model's loss can
        corrupt the P-model's carefully learned P-picking weights. The P-model
        is already excellent — there is no reason to disturb it. Freezing
        also makes the S-model's training more stable because its input
        (the P-probability vector) does not change during training.
    """
    def __init__(self, in_channels=3, base_ch=32, lstm_hidden=128,
                 lstm_layers=2, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_ch), nn.ReLU(),
        )
        self.enc1 = EncoderBlock(base_ch,     base_ch * 2, dilation=1)
        self.enc2 = EncoderBlock(base_ch * 2, base_ch * 4, dilation=2)
        self.enc3 = EncoderBlock(base_ch * 4, base_ch * 8, dilation=4)
        bottleneck_ch = base_ch * 8
        self.lstm = nn.LSTM(
            input_size=bottleneck_ch, hidden_size=lstm_hidden,
            num_layers=lstm_layers, bidirectional=True, batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_dropout = nn.Dropout(dropout)
        self.lstm_proj = nn.Sequential(
            nn.Conv1d(lstm_hidden * 2, bottleneck_ch, kernel_size=1),
            nn.BatchNorm1d(bottleneck_ch), nn.ReLU(),
        )
        self.dec1 = DecoderBlock(bottleneck_ch, base_ch * 4, base_ch * 4)
        self.dec2 = DecoderBlock(base_ch * 4,  base_ch * 2, base_ch * 2)
        self.dec3 = DecoderBlock(base_ch * 2,  base_ch,     base_ch)
        self.final_conv = nn.Sequential(
            nn.Conv1d(base_ch, base_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_ch), nn.ReLU(),
        )
        self.head_p = nn.Conv1d(base_ch, 1, kernel_size=1)
        self.head_s = nn.Conv1d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        B, C, L = x.shape
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


# ──────────────────────────────────────────────────────────────────────────────
# S-MODEL  (U-Net with in_channels=5)
# ──────────────────────────────────────────────────────────────────────────────

class CascadeSPicker(nn.Module):
    """
    Dedicated S-wave picker for the cascade pipeline.

    Architecture: identical to SeismicPickerUNet EXCEPT:
        in_channels = 5  (ZNE + P-prob + horizontal resultant)
        Only ONE output head: head_s

    WHY only one head:
        This model is specialized for S-picking only. Having a P-head would
        add a conflicting gradient signal — the model would have to simultaneously
        explain P-wave features (which it receives as a passive input channel)
        while picking S. Removing the P-head keeps the gradient signal pure.

    WHY the same U-Net depth as the P-model:
        S-wave picking requires the same receptive field as P-wave picking —
        you need to see the local onset shape AND have context about what came
        before (the P-coda). A shallower model would lose long-range context;
        a deeper one would be slower without clear benefit at this data scale.
    """

    def __init__(self, in_channels=5, base_ch=32, lstm_hidden=128,
                 lstm_layers=2, dropout=0.2):
        super().__init__()

        # ── Stem ──────────────────────────────────────────────────────────
        # Accepts 5 channels: ZNE + P-prob + horizontal resultant.
        # The first conv projects all 5 to base_ch feature maps.
        # WHY the same kernel_size=7 as the P-model stem:
        #   70ms receptive field at 100Hz captures the sharpest S-onsets.
        #   S-waves can have very sharp horizontal onsets (particularly
        #   on transverse component at close range), so a large kernel is
        #   not necessary and would smear the onset.
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(),
        )

        # ── Encoder (same dilations as P-model) ───────────────────────────
        self.enc1 = EncoderBlock(base_ch,     base_ch * 2, dilation=1)
        self.enc2 = EncoderBlock(base_ch * 2, base_ch * 4, dilation=2)
        self.enc3 = EncoderBlock(base_ch * 4, base_ch * 8, dilation=4)

        bottleneck_ch = base_ch * 8

        # ── BiLSTM bottleneck ──────────────────────────────────────────────
        # WHY keep the BiLSTM for S-picking:
        #   The S-wave model needs to understand the TEMPORAL RELATIONSHIP
        #   between the P-coda and the S-onset. An LSTM can learn that
        #   "after this type of P-wave coda, the S-wave typically looks
        #   like this." This is impossible with a pure CNN — you need memory.
        #   Bidirectionality is especially useful here: the model can look
        #   at what happens AFTER the expected S region to confirm the pick
        #   (e.g., the coda change after S).
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

        # ── Decoder ───────────────────────────────────────────────────────
        self.dec1 = DecoderBlock(bottleneck_ch, base_ch * 4, base_ch * 4)
        self.dec2 = DecoderBlock(base_ch * 4,  base_ch * 2, base_ch * 2)
        self.dec3 = DecoderBlock(base_ch * 2,  base_ch,     base_ch)
        self.final_conv = nn.Sequential(
            nn.Conv1d(base_ch, base_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(),
        )

        # ── Single S-wave head ────────────────────────────────────────────
        self.head_s = nn.Conv1d(base_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 5, 6000) — 5-channel input from build_s_input()

        Returns:
            prob_s: (B, 6000) — raw S-wave probability BEFORE masking
                    The mask is applied OUTSIDE the model (in the training
                    loop and at inference) so the model's internal probability
                    is preserved for loss computation.
        """
        B, C, L = x.shape
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

        prob_s = torch.sigmoid(self.head_s(x).squeeze(1))   # (B, 6000)
        return prob_s


# ──────────────────────────────────────────────────────────────────────────────
# LOSS
# ──────────────────────────────────────────────────────────────────────────────

def s_wave_loss(
    prob_s: torch.Tensor,
    label_s: torch.Tensor,
    sp_mask: torch.Tensor,
    pos_weight_factor: float = 10.0,
    mask_weight: float = 0.1,
) -> torch.Tensor:
    """
    S-wave loss with three components:

    1. PICKING LOSS (inside the allowed SP window):
       Weighted BCE on the masked region where S is physically plausible.
       This is the primary training signal.

    2. SUPPRESSION LOSS (outside the allowed SP window):
       Light BCE that penalises any probability mass the model places
       outside the valid SP region. Weight is low (mask_weight=0.1)
       so it doesn't dominate — its job is just to push stray predictions
       toward zero in implausible regions.

    3. SP-CONSISTENCY LOSS:
       An additional penalty proportional to the total probability mass
       that falls BEFORE t_P + min_sp. This is a soft version of the hard
       mask — it penalises early predictions more gently than the mask alone.

    WHY split the loss by mask region:
        If we just zeroed out the S-probability vector and computed BCE
        everywhere, the loss gradient would treat masked zeros as strong
        "no-S" evidence (BCE pulls predictions toward 0 strongly when the
        target is 0 and the prediction is non-zero). For traces where the
        P-model is uncertain, this would incorrectly suppress valid S-picks.
        By separating masked and unmasked regions, we can weight them
        independently and handle uncertain-P traces correctly.

    Args:
        prob_s:           (B, L) — raw S-probability from CascadeSPicker
        label_s:          (B, L) — Gaussian S-label from SeisBench (sigma=20)
        sp_mask:          (B, L) — 1 inside valid SP window, 0 outside
        pos_weight_factor: weight on positive class within the valid window
        mask_weight:      weight on the suppression loss outside the window

    Returns:
        scalar loss
    """
    pos_weight = torch.tensor(pos_weight_factor, device=prob_s.device)

    # ── 1. Picking loss (inside mask) ──────────────────────────────────────
    inside = sp_mask                          # (B, L), float
    # Weight = 1 + (pos_weight - 1) * label for each sample
    # This upweights positive class samples within the allowed window.
    pick_weight = (1 + (pos_weight - 1) * label_s) * inside
    loss_pick = F.binary_cross_entropy(prob_s, label_s, weight=pick_weight, reduction="sum")
    n_inside = inside.sum().clamp(min=1)
    loss_pick = loss_pick / n_inside

    # ── 2. Suppression loss (outside mask) ────────────────────────────────
    outside = 1.0 - sp_mask                   # (B, L)
    target_outside = torch.zeros_like(prob_s)  # target = 0 everywhere outside
    loss_suppress = F.binary_cross_entropy(
        prob_s, target_outside, weight=outside, reduction="sum"
    )
    n_outside = outside.sum().clamp(min=1)
    loss_suppress = loss_suppress / n_outside

    # ── 3. SP-consistency: soft penalty for pre-P mass ────────────────────
    # Penalise total probability mass in regions where S cannot arrive.
    # This acts as a regularizer that smoothly decays predictions near t_P.
    # WHY mean not sum: we want this to be scale-invariant to sequence length.
    loss_consistency = (prob_s * (1 - sp_mask)).mean()

    return loss_pick + mask_weight * loss_suppress + mask_weight * loss_consistency


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────

def pick_residuals(prob: torch.Tensor, label: torch.Tensor,
                   threshold: float = 0.3, tolerance: int = 50):
    """
    Mean absolute residual in samples between predicted and true pick.
    Same as previous steps — used for both P and S evaluation.
    """
    pred_pick = prob.argmax(dim=1)
    pred_max  = prob.max(dim=1).values
    true_pick = label.argmax(dim=1)
    has_pick  = label.max(dim=1).values > 0.1
    residuals = []
    for i in range(len(prob)):
        if not has_pick[i] or pred_max[i] < threshold:
            continue
        res = abs(pred_pick[i].item() - true_pick[i].item())
        if res <= tolerance:
            residuals.append(res)
    if not residuals:
        return float("nan"), 0
    return float(np.mean(residuals)), len(residuals)


def sp_interval_stats(prob_p: torch.Tensor, prob_s: torch.Tensor,
                      label_p: torch.Tensor, label_s: torch.Tensor):
    """
    Computes mean predicted S-P interval vs. true S-P interval.
    This is a physics sanity check — predicted S-P should be positive
    and within the valid Vp/Vs range for the dataset.

    If predicted S-P is systematically wrong (e.g., S picked before P),
    the mask or the SP-consistency loss needs tuning.
    """
    t_P_pred = prob_p.argmax(dim=1).float()
    t_S_pred = prob_s.argmax(dim=1).float()
    t_P_true = label_p.argmax(dim=1).float()
    t_S_true = label_s.argmax(dim=1).float()

    has_both = (label_p.max(dim=1).values > 0.1) & (label_s.max(dim=1).values > 0.1)

    pred_sp = (t_S_pred - t_P_pred)[has_both]
    true_sp = (t_S_true - t_P_true)[has_both]

    if len(pred_sp) == 0:
        return float("nan"), float("nan")

    return float(pred_sp.mean().item() / SAMPLE_RATE), float(true_sp.mean().item() / SAMPLE_RATE)


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────

def train_cascade(
    p_model: nn.Module,
    s_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int        = 50,
    learning_rate: float = 1e-3,
    device: str          = "cuda",
    checkpoint_dir: str  = "checkpoints/cascade_s",
):
    """
    Training loop for the cascade S-picker.

    Key differences from previous training loops:
        1. p_model is set to eval() and its parameters are frozen.
           WHY: we don't want the S-model's loss to distort the P-model.
        2. p_model.forward() is called inside torch.no_grad() to avoid
           storing its computation graph (saves ~2x memory).
        3. The SP mask is computed from the P-model's output each batch.
        4. The loss is s_wave_loss (mask-aware) instead of plain phase_loss.
        5. Metrics include SP-interval statistics for physics validation.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Freeze P-model permanently
    p_model = p_model.to(device).eval()
    for param in p_model.parameters():
        param.requires_grad = False

    # Only the S-model is trained
    s_model = s_model.to(device)

    optimizer = torch.optim.AdamW(s_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer, max_lr=learning_rate,
        steps_per_epoch=len(train_loader), epochs=n_epochs,
        pct_start=0.3, anneal_strategy="cos",
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_val_loss = float("inf")

    for epoch in range(1, n_epochs + 1):

        # ── Train ─────────────────────────────────────────────────────────
        s_model.train()
        train_loss = 0.0

        for batch in train_loader:
            waveform = batch["X"].to(device)               # (B, 3, 6000)
            label_p  = batch["y_p"].squeeze(1).to(device)  # (B, 6000)
            label_s  = batch["y_s"].squeeze(1).to(device)  # (B, 6000)

            # Step 1: run P-model (frozen, no gradient)
            with torch.no_grad():
                p_prob, _ = p_model(waveform)              # (B, 6000)

            # Step 2: build 5-channel S-model input
            x_s = build_s_input(waveform, p_prob)          # (B, 5, 6000)

            # Step 3: compute physics mask from P-model output
            sp_mask = build_sp_mask(p_prob).to(device)     # (B, 6000)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                prob_s = s_model(x_s)                      # (B, 6000)
                loss   = s_wave_loss(prob_s, label_s, sp_mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(s_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validate ──────────────────────────────────────────────────────
        s_model.eval()
        val_loss = 0.0
        s_res = []
        pred_sp_intervals, true_sp_intervals = [], []

        with torch.no_grad():
            for batch in val_loader:
                waveform = batch["X"].to(device)
                label_p  = batch["y_p"].squeeze(1).to(device)
                label_s  = batch["y_s"].squeeze(1).to(device)

                # P-model inference (no grad — already frozen)
                p_prob, _ = p_model(waveform)
                x_s       = build_s_input(waveform, p_prob)
                sp_mask   = build_sp_mask(p_prob).to(device)

                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    prob_s  = s_model(x_s)
                    val_loss += s_wave_loss(prob_s, label_s, sp_mask).item()

                # Apply mask to predictions for metric computation
                # WHY mask here but not in loss:
                #   The loss already accounts for the mask internally.
                #   For metrics, we want to evaluate only physically plausible
                #   picks — masking before argmax prevents the metric from
                #   counting pre-P false positives as valid picks.
                prob_s_masked = prob_s * sp_mask

                ms, _ = pick_residuals(prob_s_masked.cpu(), label_s.cpu())
                if not np.isnan(ms):
                    s_res.append(ms)

                pred_sp, true_sp = sp_interval_stats(
                    p_prob.cpu(), prob_s_masked.cpu(),
                    label_p.cpu(), label_s.cpu()
                )
                if not np.isnan(pred_sp):
                    pred_sp_intervals.append(pred_sp)
                    true_sp_intervals.append(true_sp)

        val_loss /= len(val_loader)
        print(
            f"Epoch {epoch:03d}/{n_epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"S_res={np.mean(s_res) if s_res else float('nan'):.1f}samp | "
            f"pred_SP={np.mean(pred_sp_intervals) if pred_sp_intervals else float('nan'):.2f}s | "
            f"true_SP={np.mean(true_sp_intervals) if true_sp_intervals else float('nan'):.2f}s | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "s_model_state_dict": s_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, f"{checkpoint_dir}/best_s_model.pt")

    return s_model


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE  (full cascade: P-model → mask → S-model → ensemble)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_cascade(
    p_model: nn.Module,
    s_model: nn.Module,
    waveform: np.ndarray,
    device: str = "cuda",
) -> dict:
    """
    Full two-stage cascade inference.

    Returns:
        p_sample:          argmax of P-probability (from P-model)
        s_sample:          argmax of masked S-probability (from S-model)
        p_confidence:      peak P-probability
        s_confidence:      peak S-probability after masking
        sp_interval_s:     predicted S-P interval in seconds
        prob_p:            full P-probability vector (6000,)
        prob_s_raw:        S-probability before masking (6000,)
        prob_s_masked:     S-probability after physics mask (6000,)

    WHY return both prob_s_raw and prob_s_masked:
        prob_s_raw is the model's actual belief. prob_s_masked is what you
        should use for pick reporting. Returning both lets you inspect whether
        the mask is helping or hurting — if prob_s_raw already has zero mass
        before t_P, the mask is redundant (good sign). If it has significant
        mass there, the mask is doing important work.
    """
    p_model.eval()
    s_model.eval()

    x = torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0).to(device)

    # Stage 1: P-model
    prob_p, _ = p_model(x)                          # (1, 6000)

    # Stage 2: build S-model input + mask
    x_s     = build_s_input(x, prob_p)              # (1, 5, 6000)
    sp_mask = build_sp_mask(prob_p).to(device)      # (1, 6000)

    # Stage 2: S-model
    prob_s_raw    = s_model(x_s)                    # (1, 6000)
    prob_s_masked = prob_s_raw * sp_mask             # (1, 6000)

    # Unpack to numpy
    prob_p        = prob_p[0].cpu().numpy()
    prob_s_raw    = prob_s_raw[0].cpu().numpy()
    prob_s_masked = prob_s_masked[0].cpu().numpy()
    sp_mask_np    = sp_mask[0].cpu().numpy()

    p_sample = int(prob_p.argmax())
    s_sample = int(prob_s_masked.argmax())

    return {
        "prob_p":          prob_p,
        "prob_s_raw":      prob_s_raw,
        "prob_s_masked":   prob_s_masked,
        "sp_mask":         sp_mask_np,
        "p_sample":        p_sample,
        "s_sample":        s_sample,
        "p_confidence":    float(prob_p.max()),
        "s_confidence":    float(prob_s_masked.max()),
        "sp_interval_s":   (s_sample - p_sample) / SAMPLE_RATE,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    N_EPOCHS   = 50
    LR         = 1e-3

    P_CHECKPOINT = "checkpoints/step1_unet/best_model.pt"   # your trained P-model

    # ── Load frozen P-model ────────────────────────────────────────────────
    p_model = SeismicPickerUNet(in_channels=3, base_ch=32, lstm_hidden=128,
                                lstm_layers=2, dropout=0.2)
    ckpt = torch.load(P_CHECKPOINT, map_location=DEVICE)
    p_model.load_state_dict(ckpt["model_state_dict"])
    p_model = p_model.to(DEVICE).eval()
    for param in p_model.parameters():
        param.requires_grad = False
    print(f"Loaded P-model from {P_CHECKPOINT} (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

    # ── S-model ───────────────────────────────────────────────────────────
    s_model = CascadeSPicker(in_channels=5, base_ch=32, lstm_hidden=128,
                             lstm_layers=2, dropout=0.2)
    n_params = sum(p.numel() for p in s_model.parameters() if p.requires_grad)
    print(f"S-model parameters: {n_params:,}")

    # ── Data loaders ──────────────────────────────────────────────────────
    # Note: s_sigma=20 for wider S-wave label Gaussian
    train_pipe = SeisBenchPipelineWrapper(
        dataset_name="STEAD", split="train", model_type="eqtransformer",
        transformation_sigma=10,   # P-label sigma
        s_sigma=S_LABEL_SIGMA,     # S-label sigma = 20 (wider)
    )
    val_pipe = SeisBenchPipelineWrapper(
        dataset_name="STEAD", split="dev", model_type="eqtransformer",
        transformation_sigma=10,
        s_sigma=S_LABEL_SIGMA,
    )
    train_loader = train_pipe.get_dataloader(batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
    val_loader   = val_pipe.get_dataloader(batch_size=BATCH_SIZE,   num_workers=4, shuffle=False)

    # ── Train ─────────────────────────────────────────────────────────────
    s_model = train_cascade(
        p_model=p_model,
        s_model=s_model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=N_EPOCHS,
        learning_rate=LR,
        device=DEVICE,
        checkpoint_dir="checkpoints/cascade_s",
    )

    # ── Sanity check ──────────────────────────────────────────────────────
    dummy   = np.random.randn(3, 6000).astype(np.float32)
    out     = predict_cascade(p_model, s_model, dummy, device=DEVICE)
    print(
        f"\nSanity check | "
        f"P: sample={out['p_sample']}  conf={out['p_confidence']:.3f} | "
        f"S: sample={out['s_sample']}  conf={out['s_confidence']:.3f} | "
        f"S-P={out['sp_interval_s']:.2f}s"
    )