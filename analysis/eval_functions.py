import numpy as np


def evaluate_seismic_detection(
    predictions, ground_truth, tolerance=0.1, magnitudes=None
):
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
        magnitudes:   optional list of floats (one per sample); NaN for noise/unknown.

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
            abs_errors = np.abs(diff)
            mae = np.mean(abs_errors)
        else:
            abs_errors = np.array([], dtype=float)
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
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # For tracking pure noise FPs
        noise_fp = model_fired & ~has_arrival

        # Store primary metrics
        results[f"mae_{phase}_wave"] = mae
        results[f"f1_{phase}_wave"] = f1
        results[f"precision_{phase}_wave"] = precision
        results[f"recall_{phase}_wave"] = recall
        results[f"abs_errors_{phase}_wave"] = abs_errors  # raw per-sample errors

        # Store breakdown stats for printing
        results[f"n_earthquake_{phase}"] = int(has_arrival.sum())
        results[f"n_noise_{phase}"] = int((~has_arrival).sum())
        results[f"n_noise_fp_{phase}"] = int(noise_fp.sum())
        results[f"total_fp_{phase}"] = int(fp)
        results[f"total_fn_{phase}"] = int(fn)

    _print_results(results, p_true, s_true, tolerance)

    if magnitudes is not None:
        _print_magnitude_breakdown(
            p_pred,
            s_pred,
            p_true,
            s_true,
            np.array(magnitudes, dtype=float),
            tolerance,
        )

    return results


def _compute_metrics(pred, true, tolerance):
    """Return (n, mae, precision, recall, f1) for a subset of samples."""
    has_arrival = ~np.isnan(true)
    model_fired = ~np.isnan(pred)
    n = int(has_arrival.sum())

    both_valid = has_arrival & model_fired
    mae = (
        float(np.mean(np.abs(pred[both_valid] - true[both_valid])))
        if both_valid.any()
        else float("nan")
    )

    within_tol = model_fired & has_arrival & (np.abs(pred - true) <= tolerance)
    tp = within_tol.sum()
    fp = model_fired.sum() - tp
    fn = has_arrival.sum() - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return n, mae, float(precision), float(recall), float(f1)


def _print_magnitude_breakdown(p_pred, s_pred, p_true, s_true, magnitudes, tolerance):
    """Print per-magnitude-bin evaluation metrics."""
    valid_mags = magnitudes[~np.isnan(magnitudes)]
    if len(valid_mags) == 0:
        print("  (no magnitude data available for breakdown)")
        return

    mag_min = int(np.floor(valid_mags.min()))
    mag_max = int(np.ceil(valid_mags.max()))
    bins = list(range(mag_min, mag_max + 1))

    print()
    print("=" * 80)
    print("              EVALUATION BY MAGNITUDE BIN")
    print("=" * 80)
    hdr = f"  {'Mag Bin':<10} {'N_eq':>5}  {'MAE_P':>7} {'P_F1':>7} {'P_Prec':>7} {'P_Rec':>7}  {'MAE_S':>7} {'S_F1':>7} {'S_Prec':>7} {'S_Rec':>7}"
    print(hdr)
    print("-" * 80)

    for lo in bins[:-1]:
        hi = lo + 1
        mask = (magnitudes >= lo) & (magnitudes < hi)
        if mask.sum() < 2:
            continue
        n_p, mae_p, prec_p, rec_p, f1_p = _compute_metrics(
            p_pred[mask], p_true[mask], tolerance
        )
        n_s, mae_s, prec_s, rec_s, f1_s = _compute_metrics(
            s_pred[mask], s_true[mask], tolerance
        )
        n = max(n_p, n_s)
        label = f"[{lo}, {hi})"
        mae_p_str = f"{mae_p:.4f}" if not np.isnan(mae_p) else "   N/A "
        mae_s_str = f"{mae_s:.4f}" if not np.isnan(mae_s) else "   N/A "
        print(
            f"  {label:<10} {n:>5}  {mae_p_str:>7} {f1_p:>7.4f} {prec_p:>7.4f} {rec_p:>7.4f}  {mae_s_str:>7} {f1_s:>7.4f} {prec_s:>7.4f} {rec_s:>7.4f}"
        )

    print("=" * 80)


def _print_results(results, p_true, s_true, tolerance):
    n_total = len(p_true)
    print("=" * 65)
    print("                 SEISMIC DETECTION EVALUATION")
    print("=" * 65)
    print(f"  Tolerance window    : ±{tolerance}s")
    print(f"  Total traces        : {n_total}")
    print(
        f"  Earthquake (P / S)  : {results['n_earthquake_p']} / {results['n_earthquake_s']}"
    )
    print(f"  Noise (P / S)       : {results['n_noise_p']} / {results['n_noise_s']}")
    print("-" * 65)
    print(
        f"  False Positives (P/S): {results['total_fp_p']} / {results['total_fp_s']} "
        f"(Noise only: {results['n_noise_fp_p']} / {results['n_noise_fp_s']})"
    )
    print(f"  False Negatives (P/S): {results['total_fn_p']} / {results['total_fn_s']}")
    print("-" * 65)
    print(f"  {'Metric':<30} {'P-wave':>10} {'S-wave':>10}")
    print("-" * 65)
    print(
        f"  {'MAE':<30} {results['mae_p_wave']:>10.4f} {results['mae_s_wave']:>10.4f}"
    )
    print(
        f"  {'Precision':<30} {results['precision_p_wave']:>10.4f} {results['precision_s_wave']:>10.4f}"
    )
    print(
        f"  {'Recall':<30} {results['recall_p_wave']:>10.4f} {results['recall_s_wave']:>10.4f}"
    )
    print(
        f"  {'F1-Score':<30} {results['f1_p_wave']:>10.4f} {results['f1_s_wave']:>10.4f}"
    )
    print("=" * 65)
