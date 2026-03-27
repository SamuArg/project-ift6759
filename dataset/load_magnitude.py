import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import seisbench.data as sbd

class MagnitudeDataset(Dataset):
    """
    Dataset for magnitude prediction.
    Extracts a fixed 2-second (200 sample) window starting exactly at the P-wave arrival.
    Returns (waveform_window, magnitude) pairs.
    """
    def __init__(self, sb_dataset, window_len=200, use_coords=False):
        self.sb_dataset = sb_dataset
        self.window_len = window_len
        self.use_coords = use_coords
        
        # We need to filter the dataset to only include traces with:
        # 1. A valid P-wave arrival
        # 2. A valid magnitude
        # 3. Valid latitude and longitude
        meta = self.sb_dataset.metadata
        
        p_col = "trace_p_arrival_sample" if "trace_p_arrival_sample" in meta.columns else "trace_P_arrival_sample"
        mag_cols = ["source_magnitude", "event_magnitude", "source_mag"]
        mag_col = next((c for c in mag_cols if c in meta.columns), None)
        
        lat_cols = ["station_latitude_deg", "station_latitude", "receiver_latitude", "trace_station_latitude"]
        lon_cols = ["station_longitude_deg", "station_longitude", "receiver_longitude", "trace_station_longitude"]
        lat_col = next((c for c in lat_cols if c in meta.columns), None)
        lon_col = next((c for c in lon_cols if c in meta.columns), None)
        
        if mag_col is None:
            raise ValueError("Dataset metadata does not contain a recognized magnitude column.")
            
        # Filter for rows where P-wave, magnitude are not NaN
        has_p = meta[p_col].notna().values
        has_mag = meta[mag_col].notna().values
        
        valid_mask = has_p & has_mag
        
        if self.use_coords:
            if lat_col is None or lon_col is None:
                raise ValueError("Dataset metadata does not contain recognized latitude/longitude columns.")
            has_lat = meta[lat_col].notna().values
            has_lon = meta[lon_col].notna().values
            valid_mask = valid_mask & has_lat & has_lon
            
        self.sb_dataset.filter(valid_mask)
        print(f"Filtered dataset for magnitude prediction. Retained {len(self.sb_dataset)} traces.")
        
        # Store column names for quick access during __getitem__
        self.p_col = p_col
        self.mag_col = mag_col
        self.lat_col = lat_col
        self.lon_col = lon_col
        
    def __len__(self):
        return len(self.sb_dataset)
        
    def __getitem__(self, idx):
        trace, meta = self.sb_dataset.get_sample(idx)
        
        p_arrival = int(meta[self.p_col])
        magnitude = float(meta[self.mag_col])
        
        if self.use_coords:
            lat = float(meta[self.lat_col])
            lon = float(meta[self.lon_col])
            coords = np.array([lat, lon], dtype=np.float32)
        
        # Extract the window
        # Handle cases where P-arrival + window_len exceeds trace length
        start_idx = p_arrival
        end_idx = start_idx + self.window_len
        
        if end_idx > trace.shape[1]:
            # Pad with zeros if the trace is too short
            window = np.zeros((trace.shape[0], self.window_len), dtype=np.float32)
            valid_len = trace.shape[1] - start_idx
            if valid_len > 0:
                window[:, :valid_len] = trace[:, start_idx:]
        else:
            window = trace[:, start_idx:end_idx].copy()
            
        # Optional: Apply Peak Amplitude Normalization
        # Normalize each channel by its absolute peak, or normalize globally
        max_val = np.abs(window).max(axis=1, keepdims=True)
        max_val[max_val == 0] = 1.0 # avoid division by zero
        window = window / max_val
        
        tensor_window = torch.tensor(window, dtype=torch.float32)
        tensor_mag = torch.tensor([magnitude], dtype=torch.float32)
        
        if self.use_coords:
            tensor_coords = torch.tensor(coords, dtype=torch.float32)
            return tensor_window, tensor_coords, tensor_mag
        else:
            return tensor_window, tensor_mag

def build_loaders_magnitude(dataset_name="INSTANCE", fraction=1.0, batch_size=128, window_len=200, use_coords=False):
    """
    Build train, val, test loaders for magnitude prediction.
    """
    dataset_name = dataset_name.upper()
    print(f"Loading {dataset_name} for Magnitude Prediction...")
    
    if dataset_name == "STEAD":
        dataset = sbd.STEAD(component_order="ZNE")
    elif dataset_name == "INSTANCE":
        dataset = sbd.InstanceCountsCombined(component_order="ZNE")
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
        
    train_sb = dataset.train()
    val_sb = dataset.dev()
    test_sb = dataset.test()
    
    # Subsampling if required
    if 0.0 < fraction < 1.0:
        for ds, split in [(train_sb, "train"), (val_sb, "val"), (test_sb, "test")]:
            total_events = len(ds)
            num_samples = int(total_events * fraction)
            subsample_mask = np.zeros(total_events, dtype=bool)
            random_indices = np.random.choice(total_events, num_samples, replace=False)
            subsample_mask[random_indices] = True
            ds.filter(subsample_mask)
            print(f"Subsampled {fraction*100:.1f}% of {split}: {num_samples} events selected.")
            
    train_ds = MagnitudeDataset(train_sb, window_len=window_len, use_coords=use_coords)
    val_ds = MagnitudeDataset(val_sb, window_len=window_len, use_coords=use_coords)
    test_ds = MagnitudeDataset(test_sb, window_len=window_len, use_coords=use_coords)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    return train_loader, val_loader, test_loader
