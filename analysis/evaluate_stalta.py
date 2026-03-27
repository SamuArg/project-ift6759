import sys
import os
import numpy as np
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

# Ensure we can import from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import seisbench.data as sbd
from analysis.eval_functions import evaluate_seismic_detection


def _isolated_ar_pick_worker(args):
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)

    try:
        from dataset.sta_lta import Sta_Lta

        trace, metadata, kwargs = args

        picker = Sta_Lta(
            sta=kwargs["sta"],
            lta=kwargs["lta"],
            sampling_rate=kwargs["sampling_rate"],
            low_band_pass_freq=kwargs["low_band_pass_freq"],
            upper_band_pass_freq=kwargs["upper_band_pass_freq"],
            num_ar_p=kwargs["num_ar_p"],
            num_ar_s=kwargs["num_ar_s"],
            var_window_length_p=kwargs["var_window_length_p"],
            var_window_length_s=kwargs["var_window_length_s"],
        )

        return picker.pick_trace(
            trace=trace, metadata=metadata, s_pick=True, verbose=False
        )
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stdout)
        os.close(old_stderr)


def evaluate_stalta(dataset_name="INSTANCE", fraction=1.0, tolerance=0.1):
    print(f"Loading {dataset_name} test dataset...")
    if dataset_name.upper() == "STEAD":
        sb_dataset = sbd.STEAD(component_order="ZNE").test()
    elif dataset_name.upper() == "INSTANCE":
        sb_dataset = sbd.InstanceCountsCombined(component_order="ZNE").test()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    if 0.0 < fraction < 1.0:
        total_events = len(sb_dataset)
        num_samples = int(total_events * fraction)

        subsample_mask = np.zeros(total_events, dtype=bool)
        random_indices = np.random.choice(total_events, num_samples, replace=False)
        subsample_mask[random_indices] = True

        sb_dataset.filter(subsample_mask)
        print(
            f"Subsampled {fraction*100:.1f}% of the test set: {num_samples} events selected."
        )

    predictions = []
    ground_truth = []
    magnitudes = []

    meta = sb_dataset.metadata

    p_col = (
        "trace_p_arrival_sample"
        if "trace_p_arrival_sample" in meta.columns
        else "trace_P_arrival_sample"
    )
    s_col = (
        "trace_s_arrival_sample"
        if "trace_s_arrival_sample" in meta.columns
        else "trace_S_arrival_sample"
    )

    mag_cols = ["source_magnitude", "event_magnitude", "source_mag"]
    mag_col = next((c for c in mag_cols if c in meta.columns), None)

    print(f"Running STA/LTA on {dataset_name} test set ({len(sb_dataset)} traces)...")

    pool = ProcessPoolExecutor(max_workers=1)

    for idx in tqdm(range(len(sb_dataset))):
        waveforms, metadata = sb_dataset.get_sample(idx)
        sampling_rate = metadata.get("trace_sampling_rate_hz", 100)

        # Ground truth
        true_p_sample = metadata.get(p_col, float("nan"))
        true_s_sample = metadata.get(s_col, float("nan"))

        gt_p_sec = (
            float(true_p_sample) / sampling_rate
            if not np.isnan(true_p_sample)
            else float("nan")
        )
        gt_s_sec = (
            float(true_s_sample) / sampling_rate
            if not np.isnan(true_s_sample)
            else float("nan")
        )

        mag = metadata.get(mag_col, float("nan")) if mag_col else float("nan")
        magnitudes.append(float(mag))

        ground_truth.append({"p_wave": gt_p_sec, "s_wave": gt_s_sec})

        config = {
            "sta": 0.2,
            "lta": 2.0,
            "sampling_rate": sampling_rate,
            "low_band_pass_freq": 1.0,
            "upper_band_pass_freq": 45.0,
            "num_ar_p": 2,
            "num_ar_s": 2,
            "var_window_length_p": 0.2,
            "var_window_length_s": 0.2,
        }

        try:
            future = pool.submit(
                _isolated_ar_pick_worker, (waveforms, metadata, config)
            )
            p_pick, s_pick = future.result()

            if p_pick < 0:
                p_pick = float("nan")
            if s_pick < 0:
                s_pick = float("nan")
        except BrokenProcessPool:
            pool.shutdown(wait=False)
            pool = ProcessPoolExecutor(max_workers=1)
            p_pick, s_pick = float("nan"), float("nan")
        except Exception:
            p_pick, s_pick = float("nan"), float("nan")

        predictions.append({"p_wave": float(p_pick), "s_wave": float(s_pick)})

    pool.shutdown(wait=False)

    print(f"Computing metrics for {dataset_name}...")
    metrics = evaluate_seismic_detection(
        predictions, ground_truth, tolerance=tolerance, magnitudes=magnitudes
    )

    print(f"\n--- STA/LTA Baseline Results ({dataset_name}) ---")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        elif isinstance(value, int):
            print(f"{key}: {value}")

    return metrics


if __name__ == "__main__":
    datasets = ["INSTANCE", "STEAD"]

    FRACTION = 1
    TOLERANCE = 0.1

    for ds in datasets:
        evaluate_stalta(dataset_name=ds, fraction=FRACTION, tolerance=TOLERANCE)
        print("\n\n")
