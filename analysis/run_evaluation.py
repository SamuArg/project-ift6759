"""
Standardised evaluation pipeline for seismic phase-picking models.

Usage
-----
from analysis.run_evaluation import run_evaluation

results = run_evaluation(
    model=model,                # any torch model (PhaseNet, EQTransformer, …)
    test_loader=test_loader,    # DataLoader from SeisBenchPipelineWrapper (split="test")
    sampling_rate=100,          # Hz — used to convert sample index → seconds
    confidence_threshold=0.3,   # min model output probability to count as a pick
    noise_threshold=0.1,        # min ground-truth label peak to count as an earthquake trace
    tolerance=0.1,              # pick tolerance in seconds for TP / F1
    device=None,                # torch.device; defaults to CUDA if available
)
"""

import numpy as np
import torch
from tqdm.auto import tqdm

try:
    from .eval_functions import evaluate_seismic_detection
except ImportError:
    from eval_functions import evaluate_seismic_detection


def _extract_p_s_predictions(preds):
    """
    Unpack model output into (p_prob_map, s_prob_map) tensors of shape [B, T].

    Handles:
      - PhaseNet  → tensor [B, 3, T]  (channel 0=P, 1=S, 2=noise)
      - EQTransformer → tuple (detection [B,T], p [B,T], s [B,T])
      - Generic tuple of 2  → (p [B,T], s [B,T])
    """
    if isinstance(preds, tuple):
        if len(preds) >= 3:
            # EQTransformer: (detection, p, s)
            p_map, s_map = preds[1].cpu(), preds[2].cpu()
        else:
            p_map, s_map = preds[0].cpu(), preds[1].cpu()
        # Squeeze an extra channel dim if present [B, 1, T] → [B, T]
        if p_map.ndim == 3:
            p_map = p_map.squeeze(1)
        if s_map.ndim == 3:
            s_map = s_map.squeeze(1)
    else:
        # PhaseNet / single tensor [B, 3, T]
        preds = preds.cpu()
        p_map = preds[:, 0, :]
        s_map = preds[:, 1, :]
    return p_map, s_map


def run_evaluation(
    model,
    test_loader,
    sampling_rate: int = 100,
    confidence_threshold: float = 0.3,
    noise_threshold: float = 0.1,
    tolerance: float = 0.1,
    device=None,
) -> dict:
    """
    Evaluate a seismic phase-picking model on a labelled test DataLoader.

    The loader must yield batches with keys:
        - 'X'   : waveform tensor  [B, C, T]
        - 'y_p' : P-wave Gaussian label  [B, T] or [B, 1, T]
        - 'y_s' : S-wave Gaussian label  [B, T] or [B, 1, T]

    Noise traces (no annotated arrival) are kept in the evaluation:
        - They never contribute to MSE.
        - Model picks on noise traces count as False Positives for F1.

    Parameters
    ----------
    model               : torch.nn.Module — the model to evaluate.
    test_loader         : DataLoader — from SeisBenchPipelineWrapper(split="test").
    sampling_rate       : int — waveform sampling rate in Hz (default 100).
    confidence_threshold: float — min model output to count as a pick (default 0.3).
    noise_threshold     : float — min label peak to consider a trace to have an arrival
                          (default 0.1).
    tolerance           : float — pick tolerance in seconds (default 0.1 s).
    device              : torch.device or None — defaults to GPU if available.

    Returns
    -------
    dict with keys: mse_p_wave, mse_s_wave, f1_p_wave, f1_s_wave,
                    precision_p_wave, precision_s_wave, recall_p_wave, recall_s_wave,
                    n_earthquake_p, n_earthquake_s, n_noise_p, n_noise_s,
                    n_noise_fp_p, n_noise_fp_s.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    predictions = []
    ground_truth = []

    print("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            X = batch["X"].to(device)
            y_p = batch["y_p"]
            y_s = batch["y_s"]

            # Ensure [B, T] shape for label tensors
            if y_p.ndim == 3:
                y_p = y_p[:, 0, :]
            if y_s.ndim == 3:
                y_s = y_s[:, 0, :]

            # Ground-truth peak values and positions
            p_true_max, _ = torch.max(y_p, dim=1)
            s_true_max, _ = torch.max(y_s, dim=1)
            p_true_sample = torch.argmax(y_p, dim=1)
            s_true_sample = torch.argmax(y_s, dim=1)

            # Model forward pass
            raw_preds = model(X)
            p_map, s_map = _extract_p_s_predictions(raw_preds)

            p_probs, p_pred_sample = torch.max(p_map, dim=1)
            s_probs, s_pred_sample = torch.max(s_map, dim=1)

            for i in range(X.shape[0]):
                has_p = p_true_max[i].item() >= noise_threshold
                has_s = s_true_max[i].item() >= noise_threshold

                pred_entry = {
                    # NaN when model is not confident enough (regardless of gt type)
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
                gt_entry = {
                    # NaN for noise traces (no annotated arrival)
                    "p_wave": p_true_sample[i].item() / sampling_rate if has_p else np.nan,
                    "s_wave": s_true_sample[i].item() / sampling_rate if has_s else np.nan,
                }
                predictions.append(pred_entry)
                ground_truth.append(gt_entry)

    n_eq = sum(
        1 for g in ground_truth
        if not (np.isnan(g["p_wave"]) and np.isnan(g["s_wave"]))
    )
    n_noise = len(ground_truth) - n_eq
    print(
        f"Evaluating {len(predictions)} traces "
        f"({n_eq} earthquake, {n_noise} noise)..."
    )

    results = evaluate_seismic_detection(
        predictions,
        ground_truth,
        tolerance=tolerance,
        confidence_threshold=confidence_threshold,
    )
    return results
