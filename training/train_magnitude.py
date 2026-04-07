import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

# Ensure we can import from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataset.load_magnitude import build_loaders_magnitude
from models.magnitude_model import MagnitudePredictor

def _unpack_magnitude_batch(batch: dict, device: torch.device) -> dict:
    """
    Extrait et déplace sur device toutes les features d'un batch magnitude.
    """
    out = {
        "waveform":  batch["waveform"].to(device),
        "magnitude": batch["magnitude"].to(device),
    }
    # Features optionnelles — présentes seulement si activées dans MagnitudeDataset
    for key in ("coords", "vs30", "instrument"):
        if key in batch:
            out[key] = batch[key].to(device)
    return out
 

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_mae = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        b = _unpack_magnitude_batch(batch, device)
        x   = b["waveform"]
        y   = b["magnitude"]
 
        # Construire les kwargs optionnels selon ce que le modèle attend
        call_kwargs = {}
        if "coords"     in b and getattr(model, "use_coords",     False):
            call_kwargs["coords"]     = b["coords"]
        if "vs30"       in b and getattr(model, "use_vs30",       False):
            call_kwargs["vs30"]       = b["vs30"]
        if "instrument" in b and getattr(model, "use_instrument", False):
            call_kwargs["instrument"] = b["instrument"]
       
 
        optimizer.zero_grad()
        preds = model(x, **call_kwargs)
        loss  = criterion(preds, y)
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item() * x.size(0)
        total_mae  += torch.abs(preds - y).sum().item()
 
    n_samples = len(loader.dataset)
    return total_loss / n_samples, total_mae / n_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        b = _unpack_magnitude_batch(batch, device)
        x   = b["waveform"]
        y   = b["magnitude"]
 
        call_kwargs = {}
        if "coords"     in b and getattr(model, "use_coords",     False):
            call_kwargs["coords"]     = b["coords"]
        if "vs30"       in b and getattr(model, "use_vs30",       False):
            call_kwargs["vs30"]       = b["vs30"]
        if "instrument" in b and getattr(model, "use_instrument", False):
            call_kwargs["instrument"] = b["instrument"]
 
        preds = model(x, **call_kwargs)
        loss  = criterion(preds, y)
 
        total_loss += loss.item() * x.size(0)
        total_mae  += torch.abs(preds - y).sum().item()
 
    n_samples = len(loader.dataset)
    return total_loss / n_samples, total_mae / n_samples


def train_magnitude(
    dataset_name="INSTANCE",
    epochs=20,
    batch_size=512,
    lr=1e-3,
    fraction=1.0,
    model_name="mag_predictor",
    window_len=200,
    use_coords=False,
    use_vs30=False,
    use_instrument=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_loaders_magnitude(
        dataset_name=dataset_name,
        fraction=fraction,
        batch_size=batch_size,
        window_len=window_len,
        use_coords=use_coords,
        use_vs30=use_vs30,
        use_instrument=use_instrument,
    )

    model = MagnitudePredictor(
        use_coords=use_coords,
        use_vs30=use_vs30,
        use_instrument=use_instrument,
    ).to(device)

    # Log des features actives
    active = [f for f, flag in [
        ("coords", use_coords),
        ("vs30", use_vs30),
        ("instrument", use_instrument),
    ] if flag]
    if active:
        print(f"  Features contextuelles actives : {', '.join(active)}")

    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    out_dir = os.path.join("test_outputs", "models")
    os.makedirs(out_dir, exist_ok=True)
    best_model_path = os.path.join(out_dir, f"{model_name}.pth")

    best_val_mae = float("inf")

    for epoch in range(1, epochs + 1):
        train_mse, train_mae = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_mse, val_mae = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Train MSE: {train_mse:.4f} MAE: {train_mae:.4f} | Val MSE: {val_mse:.4f} MAE: {val_mae:.4f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            print(f" => Best model saved (Val MAE: {best_val_mae:.4f})")
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best Val MAE: {best_val_mae:.4f}")
    return model


if __name__ == "__main__":
    # Small test run (fraction=0.01) to verify it works
    # train_magnitude(dataset_name="STEAD", epochs=50, fraction=1, model_name="mag_predictor_stead_200_coords", window_len=200, use_coords=True)
    # train_magnitude(dataset_name="STEAD", epochs=50, fraction=1, model_name="mag_predictor_stead_100_coords", window_len=100, use_coords=True)
    # train_magnitude(dataset_name="STEAD", epochs=50, fraction=1, model_name="mag_predictor_stead_50_coords", window_len=50, use_coords=True)
    # train_magnitude(dataset_name="STEAD", epochs=50, fraction=1, model_name="mag_predictor_stead_25_coords", window_len=25, use_coords=True)
    # train_magnitude(dataset_name="INSTANCE", epochs=50, fraction=1, model_name="mag_predictor_instance_200_coords", window_len=200, use_coords=True)

    # Exemples d'utilisation — décommenter apres que tu ais compris 
    # ─────────────────────────────────────────────────
    # STEAD — sans features contextuelles (identique à l'original)
    # train_magnitude(dataset_name="STEAD", epochs=50, fraction=1,
    #                 model_name="mag_stead_200", window_len=200)
 
    # STEAD — avec instrument uniquement (VS30 absent dans STEAD)
    # train_magnitude(dataset_name="STEAD", epochs=50, fraction=1,
    #                 model_name="mag_stead_200_instrument", window_len=200,
    #                 use_instrument=True)
 
    # INSTANCE — avec VS30 + instrument (toutes les features disponibles)
    # train_magnitude(dataset_name="INSTANCE", epochs=50, fraction=1,
    #                 model_name="mag_instance_200_vs30_instrument", window_len=200,
    #                 use_vs30=True, use_instrument=True)
 
    # Runs originaux (inchangés, VS30 et instrument désactivés par défaut)

    train_magnitude(
        dataset_name="INSTANCE",
        epochs=50,
        fraction=1,
        model_name="mag_predictor_instance_100_coords",
        window_len=100,
        use_coords=True,
    )
    train_magnitude(
        dataset_name="INSTANCE",
        epochs=50,
        fraction=1,
        model_name="mag_predictor_instance_50_coords",
        window_len=50,
        use_coords=True,
    )
    train_magnitude(
        dataset_name="INSTANCE",
        epochs=50,
        fraction=1,
        model_name="mag_predictor_instance_25_coords",
        window_len=25,
        use_coords=True,
    )
