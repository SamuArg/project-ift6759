import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import pandas as pd
import numpy as np

class CWTHDF5Dataset(Dataset):
    def __init__(self, h5_path, meta_path, split="train", fraction=1.0, sigma=10, max_distance=None):
        self.h5_path = h5_path
        self.meta = pd.read_csv(meta_path)
        
        # 1. Filter by split (SeisBench uses 'trace_dataset_split' or 'split')
        split_col = "trace_dataset_split" if "trace_dataset_split" in self.meta.columns else "split"
        self.meta = self.meta[self.meta[split_col] == split]
        
        # 2. Filter by max distance
        if max_distance is not None:
            dist_cols = ["path_hyp_distance_km", "source_distance_km"]
            dist_col = next((c for c in dist_cols if c in self.meta.columns), None)
            if dist_col:
                self.meta = self.meta[(self.meta[dist_col] <= max_distance) | self.meta[dist_col].isna()]
                
        # 3. Subsample if fraction < 1.0
        if 0.0 < fraction < 1.0:
            self.meta = self.meta.sample(frac=fraction, random_state=42)
            
        # Keep original indices to correctly reference rows in the HDF5 file
        self.indices = self.meta.index.values 
        self.meta = self.meta.reset_index(drop=True)
        
        self.sigma = sigma
        self.window_len = 3000 # 6000 samples / 2 (downsampled)

    def __len__(self):
        return len(self.meta)

    def _generate_gaussian(self, arrival_sample):
        """Generates a 1D Gaussian probability curve around the arrival."""
        y = np.zeros(self.window_len, dtype=np.float32)
        if pd.isna(arrival_sample):
            return y
            
        # IMPORTANT: Divide arrival by 2 because we decimated the data to 50 Hz!
        arrival_idx = int(arrival_sample // 2)
        
        if 0 <= arrival_idx < self.window_len:
            x = np.arange(self.window_len)
            y = np.exp(-((x - arrival_idx) ** 2) / (2 * (self.sigma ** 2)))
        return y

    def __getitem__(self, idx):
        h5_idx = self.indices[idx]
        row = self.meta.iloc[idx]
        
        # Open HDF5 file and extract the CWT
        with h5py.File(self.h5_path, 'r') as f:
            # Cast float16 back to float32 for PyTorch
            x = f['spectrograms'][h5_idx].astype(np.float32) 
            
        # Extract labels
        p_col = "trace_p_arrival_sample" if "trace_p_arrival_sample" in row else "trace_P_arrival_sample"
        s_col = "trace_s_arrival_sample" if "trace_s_arrival_sample" in row else "trace_S_arrival_sample"
        
        y_p = self._generate_gaussian(row.get(p_col, np.nan))
        y_s = self._generate_gaussian(row.get(s_col, np.nan))
        
        return torch.tensor(x), torch.tensor(y_p), torch.tensor(y_s)