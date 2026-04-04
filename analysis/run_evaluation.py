import numpy as np
import torch
from tqdm.auto import tqdm

try:
    from .eval_functions import evaluate_seismic_detection
except ImportError:
    from eval_functions import evaluate_seismic_detection


def _extract_p_s_predictions(preds):
    """Unpack model output into (p_prob_map, s_prob_map) tensors of shape [B, T] in probability space."""
    if isinstance(preds, tuple):
        p_map, s_map = (preds[1], preds[2]) if len(preds) >= 3 else (preds[0], preds[1])
        if p_map.ndim == 3:
            p_map = p_map.squeeze(1)
        if s_map.ndim == 3:
            s_map = s_map.squeeze(1)
    else:
        preds = preds.cpu()
        p_map, s_map = preds[:, 0, :], preds[:, 1, :]

    p_map = p_map.cpu().float()
    s_map = s_map.cpu().float()
    # If values are outside [0, 1] the model returned logits — convert to probs.
    if p_map.min() < 0.0 or p_map.max() > 1.0:
        p_map = torch.sigmoid(p_map)
    if s_map.min() < 0.0 or s_map.max() > 1.0:
        s_map = torch.sigmoid(s_map)
    return p_map, s_map


def run_evaluation(
    model,
    test_loader,
    sampling_rate: int = 100,
    confidence_threshold: float = 0.3,
    noise_threshold: float = 0.1,
    tolerance: float = 0.1,
    device=None,
    magnitudes=None,
) -> dict:
    """
    Evaluate a seismic phase-picking model on a labelled test DataLoader.

    Parameters
    ----------
    model                : torch.nn.Module
    test_loader          : DataLoader from SeisBenchPipelineWrapper(split="test")
    sampling_rate        : waveform sampling rate in Hz
    confidence_threshold : min model output probability to count as a pick
    noise_threshold      : min label peak to consider a trace as an earthquake
    tolerance            : pick tolerance in seconds for TP / F1
    device               : torch.device; defaults to GPU if available
    magnitudes           : optional 1-D array/list of per-sample magnitudes (NaN
                           for noise/unknown), aligned with test_loader order.
                           Pass from SeisBenchPipelineWrapper.generator.metadata.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device).eval()

    predictions = []
    ground_truth = []

    print("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            X = batch["X"].to(device)
            y_p = batch["y_p"]
            y_s = batch["y_s"]
            call_kwargs = {}
            for key in ["coords", "vs30", "instrument"]:
                val = batch.get(key, None)
                if val is not None:
                    val = val.to(device)
                    if getattr(model, f"use_{key}", False):
                        call_kwargs[key] = val

            if y_p.ndim == 3:
                y_p = y_p[:, 0, :]
            if y_s.ndim == 3:
                y_s = y_s[:, 0, :]

            p_true_max, _ = torch.max(y_p, dim=1)
            s_true_max, _ = torch.max(y_s, dim=1)
            p_true_sample = torch.argmax(y_p, dim=1)
            s_true_sample = torch.argmax(y_s, dim=1)

            p_map, s_map = _extract_p_s_predictions(model(X, **call_kwargs))

            p_probs, p_pred_sample = torch.max(p_map, dim=1)
            s_probs, s_pred_sample = torch.max(s_map, dim=1)

            for i in range(X.shape[0]):
                has_p = p_true_max[i].item() >= noise_threshold
                has_s = s_true_max[i].item() >= noise_threshold
                predictions.append(
                    {
                        "p_wave": (
                            p_pred_sample[i].item() / sampling_rate
                            if p_probs[i].item() >= confidence_threshold
                            else np.nan
                        ),
                        "s_wave": (
                            s_pred_sample[i].item() / sampling_rate
                            if s_probs[i].item() >= confidence_threshold
                            else np.nan
                        ),
                    }
                )
                ground_truth.append(
                    {
                        "p_wave": (
                            p_true_sample[i].item() / sampling_rate if has_p else np.nan
                        ),
                        "s_wave": (
                            s_true_sample[i].item() / sampling_rate if has_s else np.nan
                        ),
                    }
                )

    n_eq = sum(
        1 for g in ground_truth if not (np.isnan(g["p_wave"]) and np.isnan(g["s_wave"]))
    )
    n_noise = len(ground_truth) - n_eq
    print(
        f"Evaluating {len(predictions)} traces ({n_eq} earthquake, {n_noise} noise)..."
    )

    return evaluate_seismic_detection(
        predictions, ground_truth, tolerance=tolerance, magnitudes=magnitudes
    )
