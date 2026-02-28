import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate_seismic_detection(predictions, ground_truth, tolerance=0.1, confidence_threshold=0.3):
    """
    Evaluate seismic wave detection for P-waves and S-waves.

    Handles mixed earthquake + noise traces correctly:
      - Earthquake trace (gt is not NaN): model must pick within ±tolerance → TP.
        Outside tolerance or not picked → FP / FN. Counts toward MSE.
      - Noise trace (gt is NaN): no arrival expected. Model picks → False Positive.
        Model silent → True Negative. Never counts toward MSE.

    Args:
        predictions:  list of dicts [{'p_wave': float|nan, 's_wave': float|nan}, ...]
                      NaN means the model did not detect a pick (below confidence threshold).
        ground_truth: list of dicts [{'p_wave': float|nan, 's_wave': float|nan}, ...]
                      NaN means no annotated arrival (noise trace for that phase).
        tolerance:    float, pick tolerance window in seconds (default: 0.1 s)
        confidence_threshold: not used directly here, kept for API compatibility.

    Returns:
        dict with MSE and F1/precision/recall scores for P and S waves.
    """
    p_pred = np.array([p['p_wave'] for p in predictions], dtype=float)
    s_pred = np.array([p['s_wave'] for p in predictions], dtype=float)
    p_true = np.array([g['p_wave'] for g in ground_truth], dtype=float)
    s_true = np.array([g['s_wave'] for g in ground_truth], dtype=float)

    results = {}

    for phase, pred, true in [("p", p_pred, p_true), ("s", s_pred, s_true)]:
        has_arrival = ~np.isnan(true)       # earthquake traces for this phase
        model_fired = ~np.isnan(pred)       # model made a pick

        # ── MSE ─────────────────────────────────────────────────────────────
        # Only on earthquake traces where the model also made a pick.
        both_valid = has_arrival & model_fired
        if both_valid.any():
            diff = pred[both_valid] - true[both_valid]
            # Zero out errors that fall within the tolerance window
            errors = np.where(np.abs(diff) > tolerance, diff, 0.0)
            mse = np.mean(errors ** 2)
        else:
            mse = np.nan

        # ── F1 / Precision / Recall ──────────────────────────────────────────
        # Binary classification:
        #   true_label = 1  for earthquake traces (gt not NaN)
        #   true_label = 0  for noise traces (gt is NaN)
        #
        # predicted_label: 1 if model fired AND (it's an earthquake trace with pick
        #                  within tolerance OR it's a noise trace [FP]).
        #   - Earthquake trace + model fired + within tolerance  → TP  (pred=1, true=1)
        #   - Earthquake trace + model fired + outside tolerance → FP  (pred=1, true=1 but wrong loc)
        #     NOTE: we treat out-of-tolerance as a missed pick (FN) + spurious pick (FP).
        #     Simplification: within tolerance → TP, otherwise → FN + FP equivalent.
        #     We encode this as: predicted = 1 iff within tolerance.
        #   - Earthquake trace + model silent                    → FN  (pred=0, true=1)
        #   - Noise trace + model fired                         → FP  (pred=1, true=0)
        #   - Noise trace + model silent                        → TN  (pred=0, true=0)

        true_labels = has_arrival.astype(int)  # 1=earthquake, 0=noise

        # A "correct detection" on an earthquake trace = fired AND within tolerance
        within_tol = model_fired & has_arrival & (np.abs(pred - true) <= tolerance)
        # Noise trace firing = FP (pred=1 for true=0)
        noise_fp = model_fired & ~has_arrival

        pred_labels = np.zeros(len(true_labels), dtype=int)
        pred_labels[within_tol] = 1  # TP
        pred_labels[noise_fp] = 1    # FP (correct that pred=1 while true=0)

        f1  = f1_score(true_labels, pred_labels, zero_division=0)
        prec = precision_score(true_labels, pred_labels, zero_division=0)
        rec  = recall_score(true_labels, pred_labels, zero_division=0)

        results[f'mse_{phase}_wave']       = mse
        results[f'f1_{phase}_wave']        = f1
        results[f'precision_{phase}_wave'] = prec
        results[f'recall_{phase}_wave']    = rec

        # Extra counts for transparency
        results[f'n_earthquake_{phase}']   = int(has_arrival.sum())
        results[f'n_noise_{phase}']        = int((~has_arrival).sum())
        results[f'n_noise_fp_{phase}']     = int(noise_fp.sum())

    _print_results(results, p_pred, p_true, s_pred, s_true, tolerance)
    return results


def _print_results(results, p_pred, p_true, s_pred, s_true, tolerance):
    n_total = len(p_true)
    print("=" * 55)
    print("         SEISMIC DETECTION EVALUATION")
    print("=" * 55)
    print(f"  Tolerance window    : ±{tolerance}s")
    print(f"  Total traces        : {n_total}")
    print(f"  Earthquake (P / S)  : {results['n_earthquake_p']} / {results['n_earthquake_s']}")
    print(f"  Noise (P / S)       : {results['n_noise_p']} / {results['n_noise_s']}")
    print(f"  Noise FP  (P / S)   : {results['n_noise_fp_p']} / {results['n_noise_fp_s']}")
    print("-" * 55)
    print(f"  {'Metric':<30} {'P-wave':>10} {'S-wave':>10}")
    print("-" * 55)
    mse_p = results['mse_p_wave']
    mse_s = results['mse_s_wave']
    print(f"  {'MSE (excl. tolerance)':<30} {mse_p:>10.4f} {mse_s:>10.4f}")
    print(f"  {'Precision':<30} {results['precision_p_wave']:>10.4f} {results['precision_s_wave']:>10.4f}")
    print(f"  {'Recall':<30} {results['recall_p_wave']:>10.4f} {results['recall_s_wave']:>10.4f}")
    print(f"  {'F1-Score':<30} {results['f1_p_wave']:>10.4f} {results['f1_s_wave']:>10.4f}")
    print("=" * 55)