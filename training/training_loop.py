"""
General-purpose training loop for seismic phase-picking models.

Designed to work with:
  - Any model whose forward() accepts a (B, C, T) waveform tensor and returns
    either a (prob_p, prob_s) tuple of (B, T) tensors, or a single tensor [B, 3, T].
  - DataLoaders produced by SeisBenchPipelineWrapper (dict batches with keys
    "X", "y_p", "y_s"). Labels arrive as (B, 2, T); we use channel 0 (phase prob).
  - Custom loss functions that take (pred: Tensor, target: Tensor) → scalar.
  - Custom accuracy / metric functions with the same signature.
"""

import os
import json
import time
import copy

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _unpack_batch(batch):
    """
    Accept a SeisBench dict-batch and return
    (waveform, label_p, label_s, label_det) with labels squeezed to (B, T).

    ProbabilisticLabeller outputs (B, 2, T): channel 0 = phase probability.
    DetectionLabeller outputs (B, T) or (B, 1, T): presence of any phase.
    label_det is None if the batch does not contain 'y_det' (non-EQT pipeline).
    """
    waveform = batch["X"]
    label_p = batch["y_p"][:, 0, :]  # (B, T) — phase probability
    label_s = batch["y_s"][:, 0, :]  # (B, T)

    # y_det is only in the batch when model_type="eqtransformer" was set in
    # the pipeline (DetectionLabeller is only added for that branch).
    label_det = batch.get("y_det", None)
    if label_det is not None:
        # DetectionLabeller may return (B, T) or (B, 1, T)
        if label_det.ndim == 3:
            label_det = label_det[:, 0, :]

    return waveform, label_p, label_s, label_det


def _unpack_predictions(outputs):
    """
    Normalise model output to (prob_p, prob_s) both shaped (B, T).

    Handles:
      - (prob_p, prob_s) 2-tuple          → directly (SeismicPicker / custom)
      - (det, p, s) 3-tuple (EQTransformer) → elements 1 and 2
      - single tensor (B, 3, T) (PhaseNet) → channels 0 and 1
    """
    if isinstance(outputs, tuple):
        if len(outputs) >= 3:
            p, s = outputs[1], outputs[2]
        else:
            p, s = outputs[0], outputs[1]
        # Squeeze an extra channel dim if present: (B, 1, T) → (B, T)
        if p.ndim == 3:
            p = p.squeeze(1)
        if s.ndim == 3:
            s = s.squeeze(1)
        return p, s
    # Single tensor: assume (B, n_classes, T) with P=ch0, S=ch1
    return outputs[:, 0, :], outputs[:, 1, :]


def _get_detection_output(outputs):
    """
    Return the detection head output for EQTransformer, or None for other models.
    EQTransformer returns (det, p, s); all other models return a 2-tuple or tensor.
    """
    if isinstance(outputs, tuple) and len(outputs) >= 3:
        det = outputs[0]
        if det.ndim == 3:
            det = det.squeeze(1)
        return det
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Default loss / metric
# ─────────────────────────────────────────────────────────────────────────────


def default_phase_loss(
    outputs, targets_p, targets_s, targets_det=None, pos_weight_factor=10.0
):
    """
    Weighted BCE loss, supporting all model types:

      - SeismicPicker / PhaseNet: loss = BCE(pred_p, tgt_p) + BCE(pred_s, tgt_s)
      - EQTransformer (3-tuple):  loss += BCE(pred_det, tgt_det) when tgt_det given

    Uses binary_cross_entropy_with_logits (AMP-safe) by converting sigmoid
    probabilities back to logits via torch.logit.
    """
    pos_w = torch.tensor(pos_weight_factor, device=targets_p.device)

    def _bce(pred, target):
        # Handle (B, 1, T) shapes that may come from some models
        if pred.ndim == 3:
            pred = pred.squeeze(1)
        if target.ndim == 3:
            target = target.squeeze(1)
        weight = 1 + (pos_w - 1) * target
        logits = torch.logit(pred.float().clamp(1e-6, 1 - 1e-6))
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target.float(), weight=weight
        )

    prob_p, prob_s = _unpack_predictions(outputs)
    loss = _bce(prob_p, targets_p) + _bce(prob_s, targets_s)

    # Detection head — only active for EQTransformer
    if targets_det is not None:
        pred_det = _get_detection_output(outputs)
        if pred_det is not None:
            loss = loss + _bce(pred_det, targets_det)

    return loss


def batch_f1_score(
    outputs,
    targets_p,
    targets_s,
    confidence_threshold=0.3,
    noise_threshold=0.1,
    tolerance=10,
):
    """
    Return raw (tp, fp, fn) counts summed over P and S phases.

    Callers should accumulate these across batches and compute a single
    global F1 = 2*TP / (2*TP + FP + FN) at the end, rather than averaging
    per-batch F1 scores (which gives a biased estimate).

    For each trace:
      - Ground truth pick exists if label peak > noise_threshold
      - Model prediction exists if predicted peak > confidence_threshold
      - TP: both fire AND argmax positions within `tolerance` samples
      - FP: prediction with no nearby / no ground truth
      - FN: ground truth with no prediction or outside tolerance

    tolerance=10 samples = 0.1 s at 100 Hz (matches run_evaluation default).
    """
    prob_p, prob_s = _unpack_predictions(outputs)
    if isinstance(prob_p, torch.Tensor):
        prob_p = prob_p.float().cpu()
    if isinstance(prob_s, torch.Tensor):
        prob_s = prob_s.float().cpu()

    def _phase_counts(prob, label):
        label = label.float().cpu() if isinstance(label, torch.Tensor) else label
        pred_sample = prob.argmax(dim=1)
        pred_conf = prob.max(dim=1).values
        true_sample = label.argmax(dim=1)
        has_gt = label.max(dim=1).values > noise_threshold
        has_pred = pred_conf > confidence_threshold

        tp = fp = fn = 0
        for i in range(len(prob)):
            gt, pred = has_gt[i].item(), has_pred[i].item()
            if gt and pred:
                if abs(pred_sample[i].item() - true_sample[i].item()) <= tolerance:
                    tp += 1
                else:  # wrong location → both FP and FN
                    fp += 1
                    fn += 1
            elif pred and not gt:  # spurious pick on noise
                fp += 1
            elif gt and not pred:  # missed pick
                fn += 1
            # neither gt nor pred → TN; not counted in F1

        return tp, fp, fn

    tp_p, fp_p, fn_p = _phase_counts(prob_p, targets_p)
    tp_s, fp_s, fn_s = _phase_counts(prob_s, targets_s)
    # Return P and S counts separately so run_epoch can compute per-phase F1
    return (tp_p, fp_p, fn_p, tp_s, fp_s, fn_s)


# ─────────────────────────────────────────────────────────────────────────────
# Core epoch runner
# ─────────────────────────────────────────────────────────────────────────────


def run_epoch(
    model,
    dataloader,
    device,
    loss_fn,
    accuracy_fn,
    optimizer=None,
    scheduler=None,
    scaler=None,
    is_training=True,
    epoch_label="",
):
    """
    Run one epoch of training or validation.

    Returns (avg_loss, avg_accuracy).
    """
    model.train() if is_training else model.eval()
    use_amp = scaler is not None and scaler.is_enabled()

    total_loss = 0.0
    total_tp_p = total_fp_p = total_fn_p = 0
    total_tp_s = total_fp_s = total_fn_s = 0
    n_batches = 0

    with torch.set_grad_enabled(is_training):
        for batch in tqdm(dataloader, desc=epoch_label, leave=False):
            waveform, label_p, label_s, label_det = _unpack_batch(batch)
            waveform = waveform.to(device)
            label_p = label_p.to(device)
            label_s = label_s.to(device)
            if label_det is not None:
                label_det = label_det.to(device)

            if is_training:
                optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(waveform)
                loss = loss_fn(outputs, label_p, label_s, label_det)

            if is_training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

            with torch.no_grad():
                # Counts computed on CPU in float32 — model-type agnostic
                p_cpu = _unpack_predictions(
                    tuple(
                        o.float().cpu() if isinstance(o, torch.Tensor) else o
                        for o in outputs
                    )
                    if isinstance(outputs, tuple)
                    else outputs.float().cpu()
                )
                tp_p, fp_p, fn_p, tp_s, fp_s, fn_s = accuracy_fn(
                    p_cpu, label_p.float().cpu(), label_s.float().cpu()
                )
                total_tp_p += tp_p
                total_fp_p += fp_p
                total_fn_p += fn_p
                total_tp_s += tp_s
                total_fp_s += fp_s
                total_fn_s += fn_s

            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    denom_p = 2 * total_tp_p + total_fp_p + total_fn_p
    denom_s = 2 * total_tp_s + total_fp_s + total_fn_s
    f1_p = (2 * total_tp_p / denom_p) if denom_p > 0 else 0.0
    f1_s = (2 * total_tp_s / denom_s) if denom_s > 0 else 0.0
    return avg_loss, f1_p, f1_s


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────


def plot_metrics(
    train_losses,
    valid_losses,
    train_f1s_p,
    valid_f1s_p,
    train_f1s_s,
    valid_f1s_s,
    model_name,
    figdir,
):
    """Save loss and per-phase F1 curves to disk."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(valid_losses, label="Val Loss")
    ax1.set_title(f"{model_name} — Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(train_f1s_p, label="Train F1 P", linestyle="-", color="tab:blue")
    ax2.plot(valid_f1s_p, label="Val F1 P", linestyle="--", color="tab:blue")
    ax2.plot(train_f1s_s, label="Train F1 S", linestyle="-", color="tab:orange")
    ax2.plot(valid_f1s_s, label="Val F1 S", linestyle="--", color="tab:orange")
    ax2.set_title(f"{model_name} — Pick F1 (P & S)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1")
    ax2.legend()

    plt.tight_layout()
    os.makedirs(figdir, exist_ok=True)
    path = os.path.join(figdir, f"learning_curves_{model_name}.png")
    plt.savefig(path)
    plt.close(fig)
    print(f"Figures saved to {path}")


def train(
    model: nn.Module,
    train_set: DataLoader,
    validation_set: DataLoader,
    test_set: DataLoader,
    model_name: str,
    device=None,
    loss_fn=None,
    accuracy_fn=None,
    optimizer: optim.Optimizer = None,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    print_every: int = 1,
    logdir: str = None,
    figdir: str = None,
    modeldir: str = None,
):
    """
    Main seismic phase-picking training loop.

    Parameters
    ----------
    model          : any torch model (SeismicPicker, PhaseNet, EQTransformer…)
    train_set      : DataLoader from SeisBenchPipelineWrapper(split="train")
    validation_set : DataLoader from SeisBenchPipelineWrapper(split="dev")
    test_set       : DataLoader from SeisBenchPipelineWrapper(split="test")
    device         : torch.device or str; auto-detects GPU if None
    loss_fn        : callable(outputs, label_p, label_s) → scalar loss.
                     Defaults to weighted BCE over P and S.
    accuracy_fn    : callable(outputs, label_p, label_s) → float in [0,1].
                     Defaults to normalised pick-MAE accuracy.
    optimizer      : if None, AdamW with weight_decay=1e-4 is used.
    epochs         : number of training epochs.
    learning_rate  : peak LR for OneCycleLR (ignored if optimizer provided).
    print_every    : print epoch stats every N epochs.
    logdir         : directory for JSON training log; skipped if None.
    figdir         : directory for learning-curve plots; skipped if None.
    modeldir       : directory to save best model weights; skipped if None.

    Returns
    -------
    (model, metrics_dict)  — model loaded with best validation weights.
    """
    # ── Setup ─────────────────────────────────────────────────────────────
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device) if isinstance(device, str) else device

    if loss_fn is None:
        loss_fn = default_phase_loss
    if accuracy_fn is None:
        accuracy_fn = batch_f1_score

    model = model.to(device)

    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=len(train_set),
        epochs=epochs,
        pct_start=0.3,
        anneal_strategy="cos",
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_losses = []
    train_f1s_p, train_f1s_s = [], []
    valid_losses = []
    valid_f1s_p, valid_f1s_s = [], []
    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())

    print(f"Training {model_name} on {device} for {epochs} epochs…")
    start_time = time.time()

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):

        # ── Train ─────────────────────────────────────────────────────────
        train_loss, train_f1_p, train_f1_s = run_epoch(
            model,
            train_set,
            device,
            loss_fn,
            accuracy_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            is_training=True,
            epoch_label=f"Train {epoch}/{epochs}",
        )
        train_losses.append(train_loss)
        train_f1s_p.append(train_f1_p)
        train_f1s_s.append(train_f1_s)

        # ── Validate ──────────────────────────────────────────────────────
        val_loss, val_f1_p, val_f1_s = run_epoch(
            model,
            validation_set,
            device,
            loss_fn,
            accuracy_fn,
            is_training=False,
            epoch_label=f"Val   {epoch}/{epochs}",
        )
        valid_losses.append(val_loss)
        valid_f1s_p.append(val_f1_p)
        valid_f1s_s.append(val_f1_s)

        # ── Checkpoint ────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())

        # ── Logging ───────────────────────────────────────────────────────
        if epoch % print_every == 0 or epoch == epochs:
            lr = scheduler.get_last_lr()[0]
            tqdm.write(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_loss:.4f} f1_p={train_f1_p:.4f} f1_s={train_f1_s:.4f} | "
                f"val_loss={val_loss:.4f} f1_p={val_f1_p:.4f} f1_s={val_f1_s:.4f} | "
                f"lr={lr:.2e}"
            )

    elapsed = time.time() - start_time
    print(f"\nTraining done in {elapsed//60:.0f}m {elapsed%60:.0f}s")
    print(f"Best val loss: {best_val_loss:.4f}")

    # ── Test ──────────────────────────────────────────────────────────────
    model.load_state_dict(best_weights)
    print(f"\nEvaluating {model_name} on test set…")
    test_loss, test_f1_p, test_f1_s = run_epoch(
        model,
        test_set,
        device,
        loss_fn,
        accuracy_fn,
        is_training=False,
        epoch_label="Test",
    )
    print(f"Test loss={test_loss:.4f} | Test f1_p={test_f1_p:.4f} f1_s={test_f1_s:.4f}")

    # ── Save logs ─────────────────────────────────────────────────────────
    metrics = {
        "model": model_name,
        "epochs": epochs,
        "train_losses": train_losses,
        "valid_losses": valid_losses,
        "train_f1s_p": train_f1s_p,
        "train_f1s_s": train_f1s_s,
        "valid_f1s_p": valid_f1s_p,
        "valid_f1s_s": valid_f1s_s,
        "test_loss": test_loss,
        "test_f1_p": test_f1_p,
        "test_f1_s": test_f1_s,
    }

    if logdir is not None:
        os.makedirs(logdir, exist_ok=True)
        path = os.path.join(logdir, f"results_{model_name}.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Logs saved to {path}")

    if modeldir is not None:
        os.makedirs(modeldir, exist_ok=True)
        path = os.path.join(modeldir, f"best_{model_name}.pth")
        torch.save(best_weights, path)
        print(f"Model saved to {path}")

    if figdir is not None:
        plot_metrics(
            train_losses,
            valid_losses,
            train_f1s_p,
            valid_f1s_p,
            train_f1s_s,
            valid_f1s_s,
            model_name,
            figdir,
        )

    return model, metrics
