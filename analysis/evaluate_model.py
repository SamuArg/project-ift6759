import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seisbench.models as sbm

from models.base_lstm import SeismicPicker
from dataset.load_dataset import SeisBenchPipelineWrapper
from analysis.run_evaluation import run_evaluation

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ← edit to match what you want to evaluate
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "base_lstm"  # "base_lstm" | "phasenet" | "eqtransformer"
DATASET = "instance"  # dataset to evaluate on

# CHECKPOINT: path to a locally trained .pth file, or None to use the
# pretrained SeisBench weights already loaded by build_model().
# Examples:
#   CHECKPOINT = None                                    # pretrained SeisBench
#   CHECKPOINT = "training/test_outputs/models/best_SeismicPicker.pth"
CHECKPOINT = "training/test_outputs/models/best_SeismicPicker.pth"

CONFIDENCE_THR = 0.3  # pick confidence threshold
TOLERANCE_S = 0.1  # tolerance for TP counting (seconds)
SAMPLING_RATE = 100  # Hz

N_PLOT = 8  # number of example traces to plot
PLOT_OUT = "test_outputs/figures/sample_predictions.png"
# ─────────────────────────────────────────────────────────────────────────────


# ── Model builders ────────────────────────────────────────────────────────────


def build_model(model_name: str):
    """Reconstruct the model architecture (must match what was trained)."""
    if model_name == "base_lstm":
        return (
            SeismicPicker(
                in_channels=3,
                base_channels=64,
                lstm_hidden=128,
                lstm_layers=2,
                dropout=0.2,
            ),
            "phasenet",
        )

    elif model_name == "phasenet":
        model = sbm.PhaseNet.from_pretrained("stead")
        return model, "phasenet"

    elif model_name == "eqtransformer":
        model = sbm.EQTransformer.from_pretrained("instance")
        return model, "eqtransformer"

    raise ValueError(f"Unknown model: {model_name!r}")


def load_checkpoint(model, checkpoint):
    """
    Load weights from a local .pth file into model.

    If checkpoint is None the model is returned unchanged — this means the
    pretrained SeisBench weights loaded by build_model() are used directly.
    """
    if checkpoint is None:
        print("Using pretrained SeisBench weights (no local checkpoint).")
        return model

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint!r}.\n"
            f"Run training/test_training.py first, or set CHECKPOINT = None "
            f"to use pretrained SeisBench weights."
        )
    weights = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(weights)
    print(f"Loaded weights from {checkpoint}")
    return model


# ── Prediction helper ─────────────────────────────────────────────────────────


@torch.no_grad()
def predict_batch(model, batch, device):
    """
    Return (prob_p, prob_s) — each (B, T) on CPU — for a SeisBench dict-batch.
    Handles all three output formats.
    """
    x = batch["X"].to(device)
    out = model(x)

    if isinstance(out, tuple):
        if len(out) >= 3:  # EQTransformer (det, p, s)
            p, s = out[1], out[2]
        else:  # SeismicPicker  (p, s)
            p, s = out[0], out[1]
    else:  # PhaseNet tensor (B, 3, T)
        p, s = out[:, 0, :], out[:, 1, :]

    # squeeze any stray channel dim and move to cpu
    if p.ndim == 3:
        p = p.squeeze(1)
    if s.ndim == 3:
        s = s.squeeze(1)
    return p.float().cpu(), s.float().cpu()


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_predictions(
    model,
    plot_loader,
    device,
    n=8,
    confidence_thr=0.3,
    sampling_rate=100,
    out_path=None,
):
    """
    Plot N earthquake traces: 3-component waveform + P and S probability curves.

    Only traces with a valid ground-truth P AND S arrival are included
    (noise traces are skipped automatically).
    Use a shuffled DataLoader so the sample is representative.

    Vertical dashed lines = ground-truth arrivals.
    Horizontal dotted line = confidence threshold.
    """
    model.eval()
    collected = []

    for batch in plot_loader:
        prob_p, prob_s = predict_batch(model, batch, device)
        B = prob_p.shape[0]

        def gt_sample(label_tensor):
            """Return peak sample index; NaN if no pick (noise trace)."""
            ch0 = label_tensor[:, 0, :]  # (B, T)
            has_pick = ch0.max(dim=1).values > 0.1  # label peak above noise floor
            peak = ch0.argmax(dim=1).float()
            peak[~has_pick] = float("nan")
            return peak

        gt_p = gt_sample(batch["y_p"])  # (B,) — NaN for noise
        gt_s = gt_sample(batch["y_s"])

        for i in range(B):
            p_val = gt_p[i].item()
            s_val = gt_s[i].item()

            # Skip noise traces — only plot earthquake traces with both picks
            if np.isnan(p_val) or np.isnan(s_val):
                continue

            collected.append(
                {
                    "waveform": batch["X"][i].numpy(),  # (3, T)
                    "prob_p": prob_p[i].numpy(),  # (T,)
                    "prob_s": prob_s[i].numpy(),
                    "gt_p": p_val,
                    "gt_s": s_val,
                }
            )
            if len(collected) >= n:
                break
        if len(collected) >= n:
            break

    if not collected:
        print(
            "No earthquake traces found in the sampled batches — try increasing batch size."
        )
        return

    n_actual = len(collected)
    T = collected[0]["waveform"].shape[1]
    t = np.arange(T) / sampling_rate  # time axis in seconds

    cols = min(4, n_actual)
    rows = (n_actual + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 5, rows * 5))
    fig.suptitle(
        "Sample Predictions  (— P prob   — S prob   ‒ ‒ ground truth)",
        fontsize=13,
        y=1.01,
    )

    for idx, item in enumerate(collected):
        row_block = idx // cols
        col_pos = idx % cols

        ax_w = fig.add_subplot(rows * 2, cols, 2 * row_block * cols + col_pos + 1)
        ax_pp = fig.add_subplot(
            rows * 2, cols, (2 * row_block + 1) * cols + col_pos + 1
        )

        wv = item["waveform"]  # (3, T)

        # ── Waveform ─────────────────────────────────────────────────────
        channel_labels = ["Z", "N", "E"]
        colors_wv = ["#2d6a9f", "#3a9e6a", "#c97d2b"]
        for ch in range(min(3, wv.shape[0])):
            offset = ch * 2.5
            norm = np.abs(wv[ch]).max() + 1e-9
            ax_w.plot(
                t,
                wv[ch] / norm + offset,
                lw=0.7,
                color=colors_wv[ch],
                label=channel_labels[ch],
            )

        p_t = item["gt_p"] / sampling_rate
        s_t = item["gt_s"] / sampling_rate
        ax_w.axvline(p_t, color="#e63946", ls="--", lw=1.4, alpha=0.9, label="GT P")
        ax_w.axvline(s_t, color="#457b9d", ls="--", lw=1.4, alpha=0.9, label="GT S")
        ax_w.set_yticks([])
        ax_w.set_xlim(t[0], t[-1])
        ax_w.legend(loc="upper right", fontsize=6, ncol=5)
        ax_w.set_title(f"Trace {idx + 1}  P={p_t:.1f}s  S={s_t:.1f}s", fontsize=8)

        # ── Probability curves ────────────────────────────────────────────
        ax_pp.plot(t, item["prob_p"], color="#e63946", lw=1.3, label="P prob")
        ax_pp.plot(t, item["prob_s"], color="#457b9d", lw=1.3, label="S prob")
        ax_pp.axhline(
            confidence_thr, color="gray", ls=":", lw=0.9, label=f"thr={confidence_thr}"
        )
        ax_pp.axvline(p_t, color="#e63946", ls="--", lw=1.2, alpha=0.7)
        ax_pp.axvline(s_t, color="#457b9d", ls="--", lw=1.2, alpha=0.7)
        ax_pp.set_ylim(-0.05, 1.05)
        ax_pp.set_xlim(t[0], t[-1])
        ax_pp.set_ylabel("Prob", fontsize=8)
        ax_pp.set_xlabel("Time (s)", fontsize=8)
        ax_pp.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────
    model, pipeline_type = build_model(MODEL_NAME)
    model = load_checkpoint(model, CHECKPOINT)
    model = model.to(device).eval()

    # ── Test dataloader ───────────────────────────────────────────────────
    print(f"\nLoading {DATASET.upper()} test split…")
    test_pipe = SeisBenchPipelineWrapper(
        dataset_name=DATASET,
        split="test",
        model_type=pipeline_type,
        transformation_shape="gaussian",
        transformation_sigma=10,
    )
    test_loader = test_pipe.get_dataloader(batch_size=32, num_workers=4, shuffle=False)

    # ── Evaluation ────────────────────────────────────────────────────────
    print(f"\nRunning seismic pick evaluation…")
    run_evaluation(
        model=model,
        test_loader=test_loader,
        confidence_threshold=CONFIDENCE_THR,
        noise_threshold=0.1,
        tolerance=TOLERANCE_S,
        device=device,
    )

    # ── Shuffled loader — gives representative, varied earthquake samples ──
    # shuffle=True ensures we don't always pick the same leading traces
    # (many of which may be noise or have arrivals clustered at one position).
    plot_loader = test_pipe.get_dataloader(batch_size=32, num_workers=4, shuffle=True)

    # ── Plot ──────────────────────────────────────────────────────────────
    print(f"\nPlotting {N_PLOT} earthquake sample predictions…")
    plot_predictions(
        model=model,
        plot_loader=plot_loader,
        device=device,
        n=N_PLOT,
        confidence_thr=CONFIDENCE_THR,
        sampling_rate=SAMPLING_RATE,
        out_path=PLOT_OUT,
    )
