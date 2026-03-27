import os
import sys
import numpy as np
from sklearn.metrics import r2_score

# Ensure we can import from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataset.load_magnitude import build_loaders_magnitude


def evaluate_mean_baseline_fast(dataset_name="INSTANCE"):
    print("Building datasets...")
    # We just need the datasets, not the loaders
    train_loader, _, test_loader = build_loaders_magnitude(
        dataset_name=dataset_name, fraction=1.0, batch_size=1
    )

    train_ds = train_loader.dataset
    test_ds = test_loader.dataset

    print("Calculating mean magnitude from training set metadata...")
    train_mags = train_ds.sb_dataset.metadata[train_ds.mag_col].values
    mean_mag = np.mean(train_mags)
    print(f"Training set mean magnitude: {mean_mag:.4f}")

    print("Evaluating baseline on test set metadata...")
    test_mags = test_ds.sb_dataset.metadata[test_ds.mag_col].values

    # Predict the mean for all samples
    preds = np.full_like(test_mags, mean_mag)

    # Calculate metrics
    mae = np.mean(np.abs(preds - test_mags))
    mse = np.mean((preds - test_mags) ** 2)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_mags, preds)

    print(f"\n--- Baseline Mean Predictor Results ---")
    print(f"Samples Evaluated: {len(test_mags)}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")


if __name__ == "__main__":
    evaluate_mean_baseline_fast(dataset_name="INSTANCE")
