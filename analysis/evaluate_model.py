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
_BASE_LSTM_DEFAULTS = dict(
    CONFIDENCE_THR=0.3,
    TOLERANCE_S=0.1,
    SAMPLING_RATE=100,
    N_PLOT=0,
    PLOT_OUT=None,
    N_MISSED=0,
    MISSED_PLOT_OUT=None,
    dropout=0.2,
    lstm_layers=2,
    base_channels=64,
)

# ── 8 trained base_lstm variants ──────────────────────────────────────────────
configs = []
for dataset in ["instance", "stead"]:
    for use_coords in [False, True]:
        for lstm_hidden in [64, 128]:
            coords_str = "coords" if use_coords else "nocoords"
            name = f"base_lstm_{dataset}_h{lstm_hidden}_{coords_str}"
            configs.append({
                **_BASE_LSTM_DEFAULTS,
                "model_name":    "base_lstm",
                "dataset":       dataset,
                "model_dataset": dataset,
                "checkpoint":    f"test_outputs/models/best_{name}.pth",
                "hidden":        lstm_hidden,
                "use_coords":    use_coords,
                "label":         name,  # used for print / log
            })

# ── Pretrained PhaseNet — evaluated on both datasets ─────────────────────────
for dataset in ["instance", "stead"]:
    configs.append({
        "model_name":    "phasenet",
        "dataset":       dataset,
        "model_dataset": dataset,   # from_pretrained(dataset)
        "checkpoint":    None,
        "CONFIDENCE_THR": 0.5,
        "TOLERANCE_S":    0.1,
        "SAMPLING_RATE":  100,
        "N_PLOT":         0,
        "PLOT_OUT":       None,
        "N_MISSED":       0,
        "MISSED_PLOT_OUT": None,
        "hidden":         None,
        "dropout":        None,
        "use_coords":     False,
        "label":          f"phasenet_pretrained_{dataset}",
    })

# ── Pretrained EQTransformer — evaluated on both datasets ────────────────────
for dataset in ["instance", "stead"]:
    configs.append({
        "model_name":    "eqtransformer",
        "dataset":       dataset,
        "model_dataset": dataset,   # from_pretrained(dataset)
        "checkpoint":    None,
        "CONFIDENCE_THR": 0.3,
        "TOLERANCE_S":    0.1,
        "SAMPLING_RATE":  100,
        "N_PLOT":         0,
        "PLOT_OUT":       None,
        "N_MISSED":       0,
        "MISSED_PLOT_OUT": None,
        "hidden":         None,
        "dropout":        None,
        "use_coords":     False,
        "label":          f"eqtransformer_pretrained_{dataset}",
    })
# ─────────────────────────────────────────────────────────────────────────────


# ── Model builders ────────────────────────────────────────────────────────────


def build_model(
    model_name: str,
    checkpoint: str = None,
    model_dataset: str = None,
    lstm_hidden: int = 128,
    dropout: float = 0.2,
    base_channels: int = 64,
    lstm_layers: int = 2,
    use_coords: bool = False,
):
    """Reconstruct the model architecture (must match what was trained)."""
    is_sb_dir = checkpoint is not None and os.path.isdir(checkpoint)

    if model_name == "base_lstm":
        return (
            SeismicPicker(
                in_channels=3,
                base_channels=base_channels,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                dropout=dropout,
                use_coords=use_coords,
            ),
            "eqtransformer",  # 6000-sample window — must match train.py build_model()
        )

    elif model_name == "phasenet":
        if is_sb_dir:
            model = sbm.PhaseNet.load(checkpoint)
            print(f"Loaded SeisBench native PhaseNet from {checkpoint}")
        else:
            model = sbm.PhaseNet.from_pretrained(model_dataset)
        return model, "phasenet"

    elif model_name == "eqtransformer":
        if is_sb_dir:
            model = sbm.EQTransformer.load(checkpoint)
            print(f"Loaded SeisBench native EQTransformer from {checkpoint}")
        else:
            model = sbm.EQTransformer.from_pretrained(model_dataset)
        return model, "eqtransformer"

    elif model_name == "unet":
        from models.Upgrade1_skip_connections import SeismicPickerUNet
        return (
            SeismicPickerUNet(
                in_channels=3,
                base_ch=32,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                dropout=dropout,
            ),
            "phasenet",
        )

    elif model_name == "unet_det":
        from models.Upgrade2_detection_head import SeismicPickerUNetDet
        return (
            SeismicPickerUNetDet(
                in_channels=3,
                base_ch=32,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                dropout=dropout,
            ),
            "eqtransformer",
        )

    elif model_name == "bilstm":
        from models.bilstm import SeismicBiLSTM
        return (
            SeismicBiLSTM(
                in_channels=3,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                dropout=dropout,
            ),
            "eqtransformer",
        )

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

    if os.path.isdir(checkpoint):
        # Already natively loaded by build_model
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
    Return (prob_p, prob_s) — each (B, T) in probability space on CPU — for a
    SeisBench dict-batch.  Handles all three output formats, including the
    SeismicPicker which now returns raw logits from forward().
    """
    x = batch["X"].to(device)
    out = model(x)

    if isinstance(out, tuple):
        if len(out) >= 3:  # EQTransformer (det, p, s)
            p, s = out[1], out[2]
        else:  # SeismicPicker  (logit_p, logit_s) or (prob_p, prob_s)
            p, s = out[0], out[1]
    else:  # PhaseNet tensor (B, 3, T)
        p, s = out[:, 0, :], out[:, 1, :]

    # squeeze any stray channel dim and move to cpu
    if p.ndim == 3:
        p = p.squeeze(1)
    if s.ndim == 3:
        s = s.squeeze(1)

    p = p.float().cpu()
    s = s.float().cpu()
    # If values are outside [0, 1] the model returned logits — convert to probs.
    if p.min() < 0.0 or p.max() > 1.0:
        p = torch.sigmoid(p)
    if s.min() < 0.0 or s.max() > 1.0:
        s = torch.sigmoid(s)
    return p, s


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


# ── Missed-detection plotting ─────────────────────────────────────────────────


def plot_missed_detections(
    model,
    plot_loader,
    device,
    n=6,
    confidence_thr=0.3,
    noise_threshold=0.1,
    sampling_rate=100,
    out_path=None,
):
    """
    Find traces where a ground-truth P or S arrival exists but the model's
    probability *never* crosses `confidence_thr` (false negatives = low recall).

    Produces a figure with two sections:
      • Top rows  — missed P-wave examples
      • Bottom rows — missed S-wave examples
    Each example gets a waveform row and a probability-curve row.
    """
    model.eval()
    missed_p = []  # list of dicts: traces where GT-P exists but P not detected
    missed_s = []  # list of dicts: traces where GT-S exists but S not detected

    def gt_sample(label_tensor):
        """Return (has_pick bool, peak_sample int) for each item in the batch."""
        if label_tensor.ndim == 3:
            ch0 = label_tensor[:, 0, :]  # (B, T)
        else:
            ch0 = label_tensor  # (B, T)
        has_pick = ch0.max(dim=1).values > noise_threshold
        peak = ch0.argmax(dim=1).float()
        return has_pick, peak  # (B,), (B,)

    with torch.no_grad():
        for batch in plot_loader:
            if len(missed_p) >= n and len(missed_s) >= n:
                break

            prob_p, prob_s = predict_batch(model, batch, device)
            B = prob_p.shape[0]

            has_p, peak_p = gt_sample(batch["y_p"])
            has_s, peak_s = gt_sample(batch["y_s"])

            for i in range(B):
                wv = batch["X"][i].numpy()  # (3, T)
                pp = prob_p[i].numpy()  # (T,)
                ps = prob_s[i].numpy()  # (T,)

                # ── Missed P ──────────────────────────────────────────────
                if len(missed_p) < n and has_p[i].item():
                    if pp.max() < confidence_thr:  # model never fired for P
                        missed_p.append(
                            {
                                "waveform": wv,
                                "prob_p": pp,
                                "prob_s": ps,
                                "gt_p": peak_p[i].item(),
                                "gt_s": (
                                    peak_s[i].item()
                                    if has_s[i].item()
                                    else float("nan")
                                ),
                                "phase": "P",
                            }
                        )

                # ── Missed S ──────────────────────────────────────────────
                if len(missed_s) < n and has_s[i].item():
                    if ps.max() < confidence_thr:  # model never fired for S
                        missed_s.append(
                            {
                                "waveform": wv,
                                "prob_p": pp,
                                "prob_s": ps,
                                "gt_p": (
                                    peak_p[i].item()
                                    if has_p[i].item()
                                    else float("nan")
                                ),
                                "gt_s": peak_s[i].item(),
                                "phase": "S",
                            }
                        )

    if not missed_p and not missed_s:
        print("No missed detections found in the sampled batches — recall may be high!")
        return

    print(
        f"Missed P examples found: {len(missed_p)}, missed S examples found: {len(missed_s)}"
    )

    # ── Layout: missed-P section then missed-S section ─────────────────────────
    all_groups = []
    if missed_p:
        all_groups.append(("Missed P-wave detections", missed_p, "#e63946", "#457b9d"))
    if missed_s:
        all_groups.append(("Missed S-wave detections", missed_s, "#e63946", "#457b9d"))

    cols = min(3, max(len(g[1]) for g in all_groups))
    rows_per_group = 2  # waveform + prob
    n_groups = len(all_groups)
    total_rows = rows_per_group * n_groups

    # Each group occupies a contiguous block of rows separated by a header
    fig = plt.figure(figsize=(cols * 5, total_rows * 3 + n_groups * 0.6))
    fig.suptitle(
        f"Missed detections  (confidence thr = {confidence_thr})",
        fontsize=13,
        y=1.01,
    )

    T = None
    # We'll collect all axes per group manually
    subplot_index = 1  # 1-based index into the total grid
    grid_rows = total_rows  # total subplot rows

    for g_idx, (title, examples, col_p, col_s) in enumerate(all_groups):
        n_ex = min(len(examples), cols)
        row_w = g_idx * rows_per_group + 1  # 1-indexed waveform row
        row_p = row_w + 1  # probability row

        for ex_idx, item in enumerate(examples[:cols]):
            col_idx = ex_idx + 1  # 1-indexed column

            ax_w = fig.add_subplot(grid_rows, cols, (row_w - 1) * cols + col_idx)
            ax_pp = fig.add_subplot(grid_rows, cols, (row_p - 1) * cols + col_idx)

            wv = item["waveform"]
            if T is None:
                T = wv.shape[1]
            t = np.arange(wv.shape[1]) / sampling_rate

            # ── Waveform ─────────────────────────────────────────────────
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

            gt_p_t = item["gt_p"] / sampling_rate
            gt_s_t = item["gt_s"] / sampling_rate
            missed_phase = item["phase"]

            if not np.isnan(gt_p_t):
                ax_w.axvline(
                    gt_p_t, color="#e63946", ls="--", lw=1.4, alpha=0.9, label="GT P"
                )
            if not np.isnan(gt_s_t):
                ax_w.axvline(
                    gt_s_t, color="#457b9d", ls="--", lw=1.4, alpha=0.9, label="GT S"
                )

            ax_w.set_yticks([])
            ax_w.set_xlim(t[0], t[-1])
            ax_w.legend(loc="upper right", fontsize=6, ncol=5)
            ax_w.set_title(
                f"[{title}] ex {ex_idx+1}  GT-P={gt_p_t:.1f}s  GT-S={gt_s_t:.1f}s",
                fontsize=7,
            )

            # ── Probability curves ────────────────────────────────────────
            ax_pp.plot(t, item["prob_p"], color="#e63946", lw=1.3, label="P prob")
            ax_pp.plot(t, item["prob_s"], color="#457b9d", lw=1.3, label="S prob")
            ax_pp.axhline(
                confidence_thr,
                color="gray",
                ls=":",
                lw=0.9,
                label=f"thr={confidence_thr}",
            )
            if not np.isnan(gt_p_t):
                ax_pp.axvline(gt_p_t, color="#e63946", ls="--", lw=1.2, alpha=0.7)
            if not np.isnan(gt_s_t):
                ax_pp.axvline(gt_s_t, color="#457b9d", ls="--", lw=1.2, alpha=0.7)

            # Shade the missed phase
            missed_col = "#e63946" if missed_phase == "P" else "#457b9d"
            missed_curve = item["prob_p"] if missed_phase == "P" else item["prob_s"]
            ax_pp.fill_between(
                t,
                0,
                missed_curve,
                color=missed_col,
                alpha=0.15,
                label=f"missed {missed_phase} fill",
            )

            ax_pp.set_ylim(-0.05, 1.05)
            ax_pp.set_xlim(t[0], t[-1])
            ax_pp.set_ylabel("Prob", fontsize=8)
            ax_pp.set_xlabel("Time (s)", fontsize=8)
            ax_pp.legend(loc="upper right", fontsize=6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Missed-detection plot saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Error violin plot
# ─────────────────────────────────────────────────────────────────────────────


def plot_error_violins(
    results: dict,
    model_name: str,
    sampling_rate: int = 100,
    out_path: str = None,
):
    errors_p = results.get("abs_errors_p_wave", np.array([]))
    errors_s = results.get("abs_errors_s_wave", np.array([]))

    if len(errors_p) == 0 and len(errors_s) == 0:
        print("No errors to plot — skipping violin plot.")
        return

    # Clip zeros to a small value so log scale works
    eps = 1e-3  # 1 ms minimum
    errors_p_plot = np.clip(errors_p, eps, None) if len(errors_p) > 0 else np.array([])
    errors_s_plot = np.clip(errors_s, eps, None) if len(errors_s) > 0 else np.array([])

    data   = [d for d in [errors_p_plot, errors_s_plot] if len(d) > 0]
    labels = [l for l, d in [("P-wave", errors_p_plot), ("S-wave", errors_s_plot)] if len(d) > 0]
    colors = [c for c, d in [("#e63946", errors_p_plot), ("#457b9d", errors_s_plot)] if len(d) > 0]

    fig, ax = plt.subplots(figsize=(7, 5))

    parts = ax.violinplot(
        data,
        positions=range(len(data)),
        showmedians=True,
        showextrema=True,
        widths=0.6,
    )
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.35)
        body.set_edgecolor(color)
        body.set_linewidth(1.2)

    # Box overlay
    bp = ax.boxplot(
        data,
        positions=range(len(data)),
        widths=0.15,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # Percentile markers + annotations
    for i, (errs, label, color) in enumerate(zip(data, labels, colors)):
        median = np.median(errs)
        p95    = np.percentile(errs, 95)
        ax.scatter(i, median, color="white", zorder=5, s=40)
        ax.axhline(p95, xmin=(i + 0.05) / len(data), xmax=(i + 0.95) / len(data),
                   color=color, linewidth=1.2, linestyle="--", alpha=0.7)
        ax.text(i + 0.32, median,  f"med={median:.3f}s",  va="center", fontsize=8, color="white",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7, edgecolor="none"))
        ax.text(i + 0.32, p95,     f"p95={p95:.3f}s",     va="bottom", fontsize=7.5, color=color)

    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Absolute pick error (s)", fontsize=10)
    ax.set_title(f"{model_name} — Pick Error Distribution", fontsize=12)

    # Reference lines (text moved inside right edge to avoid x-axis labels)
    for ref_s, ref_label in [(0.1, "±0.1 s tol"), (0.5, "0.5 s"), (1.0, "1 s")]:
        ax.axhline(ref_s, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.text(len(data) - 0.55, ref_s, ref_label, va="bottom", ha="left",
                fontsize=8, color="gray")

    # Summary stats table below
    for i, (errs, label) in enumerate(zip(data, labels)):
        pct_within_tol = 100 * (errs <= 0.1).sum() / len(errs)
        pct_over_1s    = 100 * (errs >  1.0).sum() / len(errs)
        ax.text(i, ax.get_ylim()[0] * 0.6,
                f"n={len(errs):,}\n≤0.1s: {pct_within_tol:.1f}%\n>1s: {pct_over_1s:.1f}%",
                ha="center", va="top", fontsize=7.5, color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.8, edgecolor="none"))

    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Error violin plot saved to {out_path}")
    else:
        plt.show()
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main(configs):
    for config in configs:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name   = config["model_name"]
        dataset      = config["dataset"]
        model_dataset= config["model_dataset"]
        checkpoint   = config["checkpoint"]
        label        = config.get("label", f"{model_name}_{dataset}")
        CONFIDENCE_THR  = config["CONFIDENCE_THR"]
        TOLERANCE_S     = config["TOLERANCE_S"]
        SAMPLING_RATE   = config["SAMPLING_RATE"]
        N_PLOT          = config["N_PLOT"]
        PLOT_OUT        = config["PLOT_OUT"]
        N_MISSED        = config["N_MISSED"]
        MISSED_PLOT_OUT = config["MISSED_PLOT_OUT"]
        lstm_hidden  = config["hidden"]
        dropout      = config["dropout"]
        base_channels= config.get("base_channels", 64)
        lstm_layers  = config.get("lstm_layers", 2)
        use_coords   = config.get("use_coords", False)

        print("\n" + "=" * 80)
        print(f"Evaluating: {label}  |  dataset: {dataset.upper()}  |  checkpoint: {checkpoint}")
        print("=" * 80 + "\n")

        model, pipeline_type = build_model(
            model_name,
            checkpoint=checkpoint,
            model_dataset=model_dataset,
            lstm_hidden=lstm_hidden,
            dropout=dropout,
            base_channels=base_channels,
            lstm_layers=lstm_layers,
            use_coords=use_coords,
        )
        model = load_checkpoint(model, checkpoint)
        model = model.to(device).eval()
        print(
            f"Number of model parameters: {sum(p.numel() for p in model.parameters()):,}"
        )

        print(f"\nLoading {dataset.upper()} test split…")
        test_pipe = SeisBenchPipelineWrapper(
            dataset_name=dataset,
            dataset_fraction=1.0,
            split="test",
            model_type=pipeline_type,
            transformation_shape="gaussian",
            transformation_sigma=10,
            max_distance=100,
            use_coords=use_coords,
        )
        test_loader = test_pipe.get_dataloader(
            batch_size=128, num_workers=8, shuffle=False
        )

        # Extract per-trace magnitudes from dataset metadata (aligned with shuffle=False loader)
        mag_col = "source_magnitude"
        gen_meta = getattr(test_pipe.generator, "metadata", None)
        if gen_meta is not None and mag_col in gen_meta.columns:
            eval_magnitudes = gen_meta[mag_col].to_numpy(dtype=float)
            print(f"  Magnitude column found: {(~np.isnan(eval_magnitudes)).sum()} non-NaN values")
        else:
            eval_magnitudes = None
            print("  Warning: 'source_magnitude' not found in dataset metadata — skipping magnitude breakdown.")

        print(f"\nRunning seismic pick evaluation…")
        results = run_evaluation(
            model=model,
            test_loader=test_loader,
            confidence_threshold=CONFIDENCE_THR,
            noise_threshold=0.1,
            tolerance=TOLERANCE_S,
            device=device,
            magnitudes=eval_magnitudes,
        )

        violin_out = PLOT_OUT.replace(".png", f"_error_violin.png") if PLOT_OUT else None
        plot_error_violins(
            results=results,
            model_name=label,
            sampling_rate=SAMPLING_RATE,
            out_path=violin_out,
        )

        if N_PLOT > 0 and PLOT_OUT:
            plot_loader = test_pipe.get_dataloader(
                batch_size=128, num_workers=8, shuffle=True
            )
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
        if N_MISSED > 0 and MISSED_PLOT_OUT:
            if N_PLOT == 0 or not PLOT_OUT:
                plot_loader = test_pipe.get_dataloader(
                    batch_size=128, num_workers=8, shuffle=True
                )
            print(f"\nSearching for missed detections (recall failures)…")
            plot_missed_detections(
                model=model,
                plot_loader=plot_loader,
                device=device,
                n=N_MISSED,
                confidence_thr=CONFIDENCE_THR,
                noise_threshold=0.1,
                sampling_rate=SAMPLING_RATE,
                out_path=MISSED_PLOT_OUT,
            )


if __name__ == "__main__":
    main(configs)
