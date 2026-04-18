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
    batch_size=512,
    window_len=200,
    use_coords=False,
    use_vs30=False,
    use_instrument=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        # Silently skip missing models
        return None

    print(f"\nEvaluating: {os.path.basename(model_path)}")
    model = MagnitudePredictor(
        use_coords=use_coords,
        use_vs30=use_vs30,
        use_instrument=use_instrument,
    ).to(device)

    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    # We only need the test loader
    _, _, test_loader = build_loaders_magnitude(
        dataset_name=dataset_name,
        batch_size=batch_size,
        window_len=window_len,
        use_coords=use_coords,
        use_vs30=use_vs30,
        use_instrument=use_instrument,
    )

    all_preds = []
    all_targets = []
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_mae = 0.0

    for batch in test_loader:
        # Dictionary unpacking (compatible with MagnitudeDataset)
        x = batch["waveform"].to(device)
        y = batch["magnitude"].to(device)

        call_kwargs = {}
        if "coords" in batch and use_coords:
            call_kwargs["coords"] = batch["coords"].to(device)
        if "vs30" in batch and use_vs30:
            call_kwargs["vs30"] = batch["vs30"].to(device)
        if "instrument" in batch and use_instrument:
            call_kwargs["instrument"] = batch["instrument"].to(device)

        preds = model(x, **call_kwargs)

        loss = criterion(preds, y)
        total_loss += loss.item() * x.size(0)
        total_mae += torch.abs(preds - y).sum().item()

        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    n_samples = len(test_loader.dataset)
    test_mse = total_loss / n_samples
    test_mae = total_mae / n_samples

    all_preds_cat = np.concatenate(all_preds).flatten()
    all_targets_cat = np.concatenate(all_targets).flatten()
    r2 = r2_score(all_targets_cat, all_preds_cat)

    print(f"  MSE: {test_mse:.4f} | MAE: {test_mae:.4f} | R2: {r2:.4f}")

    return {
        "model": os.path.basename(model_path),
        "MSE": test_mse,
        "MAE": test_mae,
        "R2": r2,
    }


if __name__ == "__main__":
    configs = []
    for window_len in [200, 100, 50, 25]:
        for dataset in ["instance", "stead"]:
            for use_coords in [True, False]:
                for use_vs30 in [True, False]:
                    if use_vs30 and dataset == "stead":
                        continue
                    for use_instrument in [True, False]:
                        model_name = f"mag_{dataset}_{window_len}_{'coords' if use_coords else ''}_{'vs30' if use_vs30 else ''}_{'instrument' if use_instrument else ''}"
                        configs.append(
                            {
                                "dataset_name": dataset,
                                "window_len": window_len,
                                "use_coords": use_coords,
                                "use_vs30": use_vs30,
                                "use_instrument": use_instrument,
                                "model_name": model_name,
                            }
                        )

    results = []
    for config in configs:
        model_path = os.path.join(
            "test_outputs", "models", f"{config['model_name']}.pth"
        )
        res = evaluate_magnitude_model(
            model_path,
            dataset_name=config["dataset_name"],
            window_len=config["window_len"],
            use_coords=config["use_coords"],
            use_vs30=config["use_vs30"],
            use_instrument=config["use_instrument"],
        )
        if res:
            results.append(res)

    if results:
        import pandas as pd

        df = pd.DataFrame(results)
        print("\n=== Summary Table ===")
        print(df.to_string(index=False))
        df.to_csv("magnitude_final_results.csv", index=False)
