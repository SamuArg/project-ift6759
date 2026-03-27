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

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        if len(batch) == 3:
            x, coords, y = batch
            coords = coords.to(device)
        else:
            x, y = batch
            coords = None
            
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        preds = model(x, coords=coords)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        total_mae += torch.abs(preds - y).sum().item()
        
    n_samples = len(loader.dataset)
    return total_loss / n_samples, total_mae / n_samples

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
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
    use_coords=False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader, test_loader = build_loaders_magnitude(
        dataset_name=dataset_name, fraction=fraction, batch_size=batch_size, window_len=window_len, use_coords=use_coords
    )
    
    model = MagnitudePredictor(use_coords=use_coords).to(device)
    criterion = nn.MSELoss() 
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    out_dir = os.path.join("test_outputs", "models")
    os.makedirs(out_dir, exist_ok=True)
    best_model_path = os.path.join(out_dir, f"{model_name}.pth")
    
    best_val_mae = float("inf")
    
    for epoch in range(1, epochs + 1):
        train_mse, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)
        val_mse, val_mae = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train MSE: {train_mse:.4f} MAE: {train_mae:.4f} | Val MSE: {val_mse:.4f} MAE: {val_mae:.4f}")
        
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
    
    train_magnitude(dataset_name="INSTANCE", epochs=50, fraction=1, model_name="mag_predictor_instance_100_coords", window_len=100, use_coords=True)
    train_magnitude(dataset_name="INSTANCE", epochs=50, fraction=1, model_name="mag_predictor_instance_50_coords", window_len=50, use_coords=True)
    train_magnitude(dataset_name="INSTANCE", epochs=50, fraction=1, model_name="mag_predictor_instance_25_coords", window_len=25, use_coords=True)
