import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate_seismic_detection(predictions, ground_truth, tolerance=0.1):
    """
    Evaluate seismic wave detection for P-waves and S-waves.

    Earthquake trace (gt not NaN): pick within ±tolerance → TP; outside or
    silent → FN. Noise trace (gt is NaN): any pick → FP; silent → TN.
    Noise traces never contribute to MSE.

    Args:
        predictions:  list of dicts [{'p_wave': float|nan, 's_wave': float|nan}]
                      NaN means the model did not detect a pick.
        ground_truth: list of dicts [{'p_wave': float|nan, 's_wave': float|nan}]
                      NaN means no annotated arrival (noise trace).
        tolerance:    pick tolerance window in seconds (default 0.1 s)

    Returns:
        dict with MSE and F1/precision/recall scores for P and S waves.
    """
    p_pred = np.array([p['p_wave'] for p in predictions], dtype=float)
    s_pred = np.array([p['s_wave'] for p in predictions], dtype=float)
    p_true = np.array([g['p_wave'] for g in ground_truth], dtype=float)
    s_true = np.array([g['s_wave'] for g in ground_truth], dtype=float)

    results = {}

    for phase, pred, true in [("p", p_pred, p_true), ("s", s_pred, s_true)]:
        has_arrival = ~np.isnan(true)
        model_fired = ~np.isnan(pred)

        both_valid = has_arrival & model_fired
        if both_valid.any():
            diff   = pred[both_valid] - true[both_valid]
            errors = np.where(np.abs(diff) > tolerance, diff, 0.0)
            mse    = np.mean(errors ** 2)
        else:
            mse = np.nan

        true_labels = has_arrival.astype(int)
        within_tol  = model_fired & has_arrival & (np.abs(pred - true) <= tolerance)
        noise_fp    = model_fired & ~has_arrival

        pred_labels = np.zeros(len(true_labels), dtype=int)
        pred_labels[within_tol] = 1
        pred_labels[noise_fp]   = 1

        results[f'mse_{phase}_wave']       = mse
        results[f'f1_{phase}_wave']        = f1_score(true_labels, pred_labels, zero_division=0)
        results[f'precision_{phase}_wave'] = precision_score(true_labels, pred_labels, zero_division=0)
        results[f'recall_{phase}_wave']    = recall_score(true_labels, pred_labels, zero_division=0)
        results[f'n_earthquake_{phase}']   = int(has_arrival.sum())
        results[f'n_noise_{phase}']        = int((~has_arrival).sum())
        results[f'n_noise_fp_{phase}']     = int(noise_fp.sum())

    _print_results(results, p_true, s_true, tolerance)
    return results


def _print_results(results, p_true, s_true, tolerance):
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
    print(f"  {'MSE (excl. tolerance)':<30} {results['mse_p_wave']:>10.4f} {results['mse_s_wave']:>10.4f}")
    print(f"  {'Precision':<30} {results['precision_p_wave']:>10.4f} {results['precision_s_wave']:>10.4f}")
    print(f"  {'Recall':<30} {results['recall_p_wave']:>10.4f} {results['recall_s_wave']:>10.4f}")
    print(f"  {'F1-Score':<30} {results['f1_p_wave']:>10.4f} {results['f1_s_wave']:>10.4f}")
    print("=" * 55)