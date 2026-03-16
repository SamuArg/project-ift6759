import numpy as np


def evaluate_seismic_detection(predictions, ground_truth, tolerance=0.1):
    """
    Evaluate seismic wave detection for P-waves and S-waves.

    - True Positive (TP): Model pick is within ±tolerance of the true arrival.
    - False Positive (FP): Model fired, but either no true arrival exists (noise) 
                           OR the pick is outside the tolerance window.
    - False Negative (FN): True arrival exists, but model either didn't fire 
                           OR fired outside the tolerance window.

    Args:
        predictions:  list of dicts [{'p_wave': float|nan, 's_wave': float|nan}]
                      NaN means the model did not detect a pick.
        ground_truth: list of dicts [{'p_wave': float|nan, 's_wave': float|nan}]
                      NaN means no annotated arrival (noise trace).
        tolerance:    pick tolerance window in seconds (default 0.1 s).
                      Only used for the F1/precision/recall binary labels;
                      MAE is computed on raw errors regardless of tolerance.

    Returns:
        dict with MAE and F1/precision/recall scores for P and S waves.
    """
    p_pred = np.array([p["p_wave"] for p in predictions], dtype=float)
    s_pred = np.array([p["s_wave"] for p in predictions], dtype=float)
    p_true = np.array([g["p_wave"] for g in ground_truth], dtype=float)
    s_true = np.array([g["s_wave"] for g in ground_truth], dtype=float)

    results = {}

    for phase, pred, true in [("p", p_pred, p_true), ("s", s_pred, s_true)]:
        has_arrival = ~np.isnan(true)
        model_fired = ~np.isnan(pred)

        # MAE is computed on raw errors whenever both model and GT have a pick
        both_valid = has_arrival & model_fired
        if both_valid.any():
            diff = pred[both_valid] - true[both_valid]
            mae = np.mean(np.abs(diff))
        else:
            mae = np.nan

        # Pick-level detection logic
        within_tol = model_fired & has_arrival & (np.abs(pred - true) <= tolerance)
        
        # Calculate TP, FP, FN manually
        tp = within_tol.sum()
        fp = model_fired.sum() - tp
        fn = has_arrival.sum() - tp

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # For tracking pure noise FPs 
        noise_fp = model_fired & ~has_arrival

        # Store primary metrics
        results[f"mae_{phase}_wave"] = mae
        results[f"f1_{phase}_wave"] = f1
        results[f"precision_{phase}_wave"] = precision
        results[f"recall_{phase}_wave"] = recall
        
        # Store breakdown stats for printing
        results[f"n_earthquake_{phase}"] = int(has_arrival.sum())
        results[f"n_noise_{phase}"] = int((~has_arrival).sum())
        results[f"n_noise_fp_{phase}"] = int(noise_fp.sum())
        results[f"total_fp_{phase}"] = int(fp)
        results[f"total_fn_{phase}"] = int(fn)

    _print_results(results, p_true, s_true, tolerance)
    return results


def _print_results(results, p_true, s_true, tolerance):
    n_total = len(p_true)
    print("=" * 65)
    print("                 SEISMIC DETECTION EVALUATION")
    print("=" * 65)
    print(f"  Tolerance window    : ±{tolerance}s")
    print(f"  Total traces        : {n_total}")
    print(f"  Earthquake (P / S)  : {results['n_earthquake_p']} / {results['n_earthquake_s']}")
    print(f"  Noise (P / S)       : {results['n_noise_p']} / {results['n_noise_s']}")
    print("-" * 65)
    print(f"  False Positives (P/S): {results['total_fp_p']} / {results['total_fp_s']} "
          f"(Noise only: {results['n_noise_fp_p']} / {results['n_noise_fp_s']})")
    print(f"  False Negatives (P/S): {results['total_fn_p']} / {results['total_fn_s']}")
    print("-" * 65)
    print(f"  {'Metric':<30} {'P-wave':>10} {'S-wave':>10}")
    print("-" * 65)
    print(f"  {'MAE':<30} {results['mae_p_wave']:>10.4f} {results['mae_s_wave']:>10.4f}")
    print(f"  {'Precision':<30} {results['precision_p_wave']:>10.4f} {results['precision_s_wave']:>10.4f}")
    print(f"  {'Recall':<30} {results['recall_p_wave']:>10.4f} {results['recall_s_wave']:>10.4f}")
    print(f"  {'F1-Score':<30} {results['f1_p_wave']:>10.4f} {results['f1_s_wave']:>10.4f}")
    print("=" * 65)