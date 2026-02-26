import numpy as np
from sklearn.metrics import f1_score

def evaluate_seismic_detection(predictions, ground_truth, tolerance=0.1):
    """
    Evaluate seismic wave detection for P-waves and S-waves.
    
    Args:
        predictions: list of dicts [{'p_wave': float, 's_wave': float}, ...]
        ground_truth: list of dicts [{'p_wave': float, 's_wave': float}, ...]
        tolerance: float, tolerance window in seconds (default: 0.1)
    
    Returns:
        dict with MSE and F1 scores for P and S waves
    """
    
    p_pred = np.array([p['p_wave'] for p in predictions])
    s_pred = np.array([p['s_wave'] for p in predictions])
    p_true = np.array([g['p_wave'] for g in ground_truth])
    s_true = np.array([g['s_wave'] for g in ground_truth])
    
    # --- MSE (only on errors outside tolerance window) ---
    p_diff = p_pred - p_true
    s_diff = s_pred - s_true
    
    p_errors = np.where(np.abs(p_diff) > tolerance, p_diff, 0.0)
    s_errors = np.where(np.abs(s_diff) > tolerance, s_diff, 0.0)
    
    mse_p = np.mean(p_errors ** 2)
    mse_s = np.mean(s_errors ** 2)
    
    # --- F1 Score (detection within tolerance = TP) ---
    p_correct = (np.abs(p_diff) <= tolerance).astype(int)
    s_correct = (np.abs(s_diff) <= tolerance).astype(int)
    
    # All ground truth = 1 (positive class), correct detection = 1, missed/wrong = 0
    p_true_labels = np.ones(len(p_true), dtype=int)
    s_true_labels = np.ones(len(s_true), dtype=int)
    
    f1_p = f1_score(p_true_labels, p_correct, zero_division=0)
    f1_s = f1_score(s_true_labels, s_correct, zero_division=0)
    
    results = {
        'mse_p_wave': mse_p,
        'mse_s_wave': mse_s,
        'f1_p_wave':  f1_p,
        'f1_s_wave':  f1_s,
    }
    
    _print_results(results, p_diff, s_diff, tolerance)
    return results

def _print_results(results, p_diff, s_diff, tolerance):
    print("=" * 45)
    print("       SEISMIC DETECTION EVALUATION")
    print("=" * 45)
    print(f"  Tolerance window : ±{tolerance}s")
    print(f"  Samples          : {len(p_diff)}")
    print("-" * 45)
    print(f"  {'Metric':<25} {'P-wave':>8} {'S-wave':>8}")
    print("-" * 45)
    print(f"  {'MSE (with tolerance)':<25} {results['mse_p_wave']:>8.4f} {results['mse_s_wave']:>8.4f}")
    print(f"  {'F1-Score':<25} {results['f1_p_wave']:>8.4f} {results['f1_s_wave']:>8.4f}")
    print("=" * 45)