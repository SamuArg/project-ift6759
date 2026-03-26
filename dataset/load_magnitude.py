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
    def __init__(self, sb_dataset, window_len=200):
        self.sb_dataset = sb_dataset
        self.window_len = window_len
        
        # We need to filter the dataset to only include traces with:
        # 1. A valid P-wave arrival
        # 2. A valid magnitude
        meta = self.sb_dataset.metadata
        
        p_col = "trace_p_arrival_sample" if "trace_p_arrival_sample" in meta.columns else "trace_P_arrival_sample"
        mag_cols = ["source_magnitude", "event_magnitude", "source_mag"]
        mag_col = next((c for c in mag_cols if c in meta.columns), None)
        
        if mag_col is None:
            raise ValueError("Dataset metadata does not contain a recognized magnitude column.")
            
        # Filter for rows where both P-wave and magnitude are not NaN
        has_p = meta[p_col].notna().values
        has_mag = meta[mag_col].notna().values
        
        # Additionally, make sure the trace is long enough to extract the window
        # trace_length is usually in meta, if not assume we can index up to the required length
        valid_mask = has_p & has_mag
        
        self.sb_dataset.filter(valid_mask)
        print(f"Filtered dataset for magnitude prediction. Retained {len(self.sb_dataset)} traces.")
        
        # Store column names for quick access during __getitem__
        self.p_col = p_col
        self.mag_col = mag_col
        
    def __len__(self):
        return len(self.sb_dataset)
        
    def __getitem__(self, idx):
        trace, meta = self.sb_dataset.get_sample(idx)
        
        p_arrival = int(meta[self.p_col])
        magnitude = float(meta[self.mag_col])
        
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
        
        return torch.tensor(window, dtype=torch.float32), torch.tensor([magnitude], dtype=torch.float32)

def build_loaders_magnitude(dataset_name="INSTANCE", fraction=1.0, batch_size=128, window_len=200):
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
            
    train_ds = MagnitudeDataset(train_sb, window_len=window_len)
    val_ds = MagnitudeDataset(val_sb, window_len=window_len)
    test_ds = MagnitudeDataset(test_sb, window_len=window_len)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    return train_loader, val_loader, test_loader
