import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Ensure we can import from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataset.load_magnitude import build_loaders_magnitude
from models.magnitude_model import MagnitudePredictor


@torch.no_grad()
def evaluate_magnitude_model(
    model_path,
    dataset_name="INSTANCE",
    batch_size=128,
    fraction=1.0,
    window_len=200,
    use_coords=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {model_path} onto {device}...")

    model = MagnitudePredictor(use_coords=use_coords).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: {model_path} not found. Using randomly initialized model.")

    model.eval()

    # We only need the test loader
    _, _, test_loader = build_loaders_magnitude(
        dataset_name=dataset_name,
        fraction=fraction,
        batch_size=batch_size,
        window_len=window_len,
        use_coords=use_coords,
    )

    all_preds = []
    all_targets = []

    criterion = nn.MSELoss()
    total_loss = 0.0
    total_mae = 0.0

    print("Evaluating on test set...")
    for batch in test_loader:
        if len(batch) == 3:
            x, coords, y = batch
            coords = coords.to(device)
        else:
            x, y = batch
            coords = None

        x, y = x.to(device), y.to(device)
        preds = model(x, coords=coords)

        loss = criterion(preds, y)
        total_loss += loss.item() * x.size(0)
        total_mae += torch.abs(preds - y).sum().item()

        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    n_samples = len(test_loader.dataset)
    test_mse = total_loss / n_samples
    test_mae = total_mae / n_samples
    test_rmse = np.sqrt(test_mse)

    all_preds_cat = np.concatenate(all_preds).flatten()
    all_targets_cat = np.concatenate(all_targets).flatten()
    r2 = r2_score(all_targets_cat, all_preds_cat)

    print(f"\n--- Magnitude Prediction Test Results ---")
    print(f"Samples Evaluated: {n_samples}")
    print(f"Mean Squared Error (MSE): {test_mse:.4f}")
    print(f"Mean Absolute Error (MAE): {test_mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {test_rmse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

    # Plotting
    os.makedirs(os.path.join("test_outputs", "figures"), exist_ok=True)
    plot_path = os.path.join("test_outputs", "figures", "magnitude_scatter.png")

    plt.figure(figsize=(8, 8))
    plt.scatter(all_targets_cat, all_preds_cat, alpha=0.3, s=10)
    plt.plot(
        [all_targets_cat.min(), all_targets_cat.max()],
        [all_targets_cat.min(), all_targets_cat.max()],
        "r--",
        lw=2,
        label="Ideal",
    )
    plt.xlabel("True Magnitude")
    plt.ylabel("Predicted Magnitude")
    plt.title(f"True vs Predicted Magnitude\nMAE: {test_mae:.2f} | R²: {r2:.2f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    print(f"Scatter plot saved to {plot_path}")


if __name__ == "__main__":
    # Test evaluation
    datasets = ["INSTANCE", "STEAD"]
    input_sizes = [25, 50, 100, 200]
    for dataset in datasets:
        for size in input_sizes:
            print(f"Testing {dataset} with input size {size}")
            model_path = os.path.join(
                "test_outputs",
                "models",
                f"mag_predictor_{dataset.lower()}_{size}_coords.pth",
            )
            evaluate_magnitude_model(
                model_path,
                dataset_name=dataset,
                fraction=1,
                window_len=size,
                use_coords=True,
            )
