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


def _unpack_batch(batch):
    """Unpack a SeisBench dict-batch into (waveform, label_p, label_s, label_det, coords)."""
    waveform = batch["X"]
    label_p = batch["y_p"][:, 0, :]
    label_s = batch["y_s"][:, 0, :]
    coords = batch.get("coords", None)
    vs30       = batch.get("vs30", None)         
    instrument = batch.get("instrument", None)   

    label_det = batch.get("y_det", None)
    if label_det is not None:
        if label_det.ndim == 3:
            label_det = label_det[:, 0, :]

    return waveform, label_p, label_s, label_det, coords, vs30, instrument


def _unpack_predictions(outputs):
    """Normalize model output to (prob_p, prob_s) both shaped (B, T)."""
    if isinstance(outputs, tuple):
        if len(outputs) >= 3:
            p, s = outputs[1], outputs[2]
        else:
            p, s = outputs[0], outputs[1]
        if p.ndim == 3:
            p = p.squeeze(1)
        if s.ndim == 3:
            s = s.squeeze(1)
        return p, s
    return outputs[:, 0, :], outputs[:, 1, :]


def _get_detection_output(outputs):
    """Return the detection head output for EQTransformer, or None for other models."""
    if isinstance(outputs, tuple) and len(outputs) >= 3:
        det = outputs[0]
        if det.ndim == 3:
            det = det.squeeze(1)
        return det
    return None


def default_phase_loss(
    outputs, targets_p, targets_s, targets_det=None, pos_weight_factor=10.0
):
    """Weighted BCE loss supporting SeismicPicker (logits) and SeisBench models (probs)."""
    pos_w = torch.tensor(pos_weight_factor, device=targets_p.device)

    def _bce(pred, target):
        if pred.ndim == 3:
            pred = pred.squeeze(1)
        if target.ndim == 3:
            target = target.squeeze(1)
        pred = pred.float()
        target = target.float()
        weight = 1 + (pos_w - 1) * target
        if pred.min() < 0.0 or pred.max() > 1.0:
            logits = pred
        else:
            logits = torch.logit(pred.clamp(1e-6, 1 - 1e-6))
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, weight=weight
        )

    prob_p, prob_s = _unpack_predictions(outputs)
    loss = _bce(prob_p, targets_p) + 2 * _bce(prob_s, targets_s)

    if targets_det is not None:
        pred_det = _get_detection_output(outputs)
        if pred_det is not None:
            targets_for_det = targets_det
            if pred_det.ndim == 1 and targets_for_det.ndim == 2:
                targets_for_det = (targets_for_det.max(dim=-1).values > 0.5).float()
            loss = loss + _bce(pred_det, targets_for_det)

    return loss


def batch_f1_score(
    outputs,
    targets_p,
    targets_s,
    confidence_threshold=0.3,
    noise_threshold=0.1,
    tolerance=10,
):
    """Return raw (tp, fp, fn) counts for P and S. Accumulate across batches before computing F1."""
    prob_p, prob_s = _unpack_predictions(outputs)
    if isinstance(prob_p, torch.Tensor):
        prob_p = prob_p.float().cpu()
    if isinstance(prob_s, torch.Tensor):
        prob_s = prob_s.float().cpu()

    def _phase_counts(prob, label):
        label = label.float().cpu() if isinstance(label, torch.Tensor) else label
        pred_sample = prob.argmax(dim=1)
        prob_for_thresh = torch.sigmoid(prob) if prob.min() < 0.0 else prob
        pred_conf = prob_for_thresh.max(dim=1).values
        true_sample = label.argmax(dim=1)
        has_gt = label.max(dim=1).values > noise_threshold
        has_pred = pred_conf > confidence_threshold

        tp = fp = fn = 0
        for i in range(len(prob)):
            gt, pred = has_gt[i].item(), has_pred[i].item()
            if gt and pred:
                if abs(pred_sample[i].item() - true_sample[i].item()) <= tolerance:
                    tp += 1
                else:
                    fp += 1
                    fn += 1
            elif pred and not gt:
                fp += 1
            elif gt and not pred:
                fn += 1

        return tp, fp, fn

    tp_p, fp_p, fn_p = _phase_counts(prob_p, targets_p)
    tp_s, fp_s, fn_s = _phase_counts(prob_s, targets_s)
    return (tp_p, fp_p, fn_p, tp_s, fp_s, fn_s)


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
    """Run one epoch of training or validation. Returns (avg_loss, f1_p, f1_s)."""
    model.train() if is_training else model.eval()
    use_amp = scaler is not None and scaler.is_enabled()

    total_loss = 0.0
    total_tp_p = total_fp_p = total_fn_p = 0
    total_tp_s = total_fp_s = total_fn_s = 0
    n_batches = 0

    with torch.set_grad_enabled(is_training):
        for batch in tqdm(dataloader, desc=epoch_label, leave=False):
            waveform, label_p, label_s, label_det, coords, vs30, instrument = (
                _unpack_batch(batch)
            )
            waveform = waveform.to(device)
            label_p = label_p.to(device)
            label_s = label_s.to(device)
            if label_det is not None:
                label_det = label_det.to(device)
            if coords is not None:
                coords = coords.to(device)
            if vs30 is not None:
                vs30 = vs30.to(device)
            if instrument is not None:
                instrument = instrument.to(device)

            if is_training:
                optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                call_kwargs = {}
                if coords is not None and getattr(model, "use_coords", False):
                    call_kwargs["coords"] = coords
                if vs30 is not None and getattr(model, "use_vs30", False):
                    call_kwargs["vs30"] = vs30
                if instrument is not None and getattr(model, "use_instrument", False):
                    call_kwargs["instrument"] = instrument
 
                outputs = model(waveform, **call_kwargs)
                loss = loss_fn(outputs, label_p, label_s, label_det)

            if is_training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None and scaler.get_scale() == scale_before:
                    scheduler.step()

            with torch.no_grad():
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
    use_amp: bool = None,
):
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

    if use_amp is None:
        use_amp = device.type == "cuda"
    if not use_amp and device.type == "cuda":
        print("AMP disabled for this model (float16 incompatible).")
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())

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

        if hasattr(model, "save"):
            sb_path = os.path.join(modeldir, f"seisbench_{model_name}")
            model.save(sb_path)
            print(f"SeisBench native model saved to {sb_path}")

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
