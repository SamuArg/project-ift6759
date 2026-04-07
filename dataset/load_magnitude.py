import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import seisbench.data as sbd

# need this for new metadata
from dataset.load_dataset import N_INSTRUMENT_CLASSES, INSTRUMENT_CODE_TO_IDX

class MagnitudeDataset(Dataset):
    """
    Dataset for magnitude prediction.
    Extracts a fixed 2-second (200 sample) window starting exactly at the P-wave arrival.
    Returns (waveform_window, magnitude) pairs.
    """

    # Noms de colonnes VS30 (INSTANCE uniquement)
    VS30_COL = "station_vs_30_mps"
 
    # Ordre de priorité pour la colonne instrument :
    # station_channels (INSTANCE) avant trace_channel (STEAD)
    CHANNEL_COLS = ["station_channels", "trace_channel"]

    def __init__(self, sb_dataset, window_len=200, use_coords=False, use_vs30=False, use_instrument=False):
        self.sb_dataset = sb_dataset
        self.window_len = window_len
        self.use_coords = use_coords
        self.use_vs30     = use_vs30
        self.use_instrument = use_instrument

        # We need to filter the dataset to only include traces with:
        # 1. A valid P-wave arrival
        # 2. A valid magnitude
        # 3. Valid latitude and longitude
        meta = self.sb_dataset.metadata

        p_col = (
            "trace_p_arrival_sample"
            if "trace_p_arrival_sample" in meta.columns
            else "trace_P_arrival_sample"
        )
        mag_cols = ["source_magnitude", "event_magnitude", "source_mag"]
        mag_col = next((c for c in mag_cols if c in meta.columns), None)

        lat_cols = [
            "station_latitude_deg",
            "station_latitude",
            "receiver_latitude",
            "trace_station_latitude",
        ]
        lon_cols = [
            "station_longitude_deg",
            "station_longitude",
            "receiver_longitude",
            "trace_station_longitude",
        ]
        lat_col = next((c for c in lat_cols if c in meta.columns), None)
        lon_col = next((c for c in lon_cols if c in meta.columns), None)

        if mag_col is None:
            raise ValueError(
                "Dataset metadata does not contain a recognized magnitude column."
            )

        lat_col = next((c for c in lat_cols if c in meta.columns), None)
        lon_col = next((c for c in lon_cols if c in meta.columns), None)
 
        if self.use_coords and (lat_col is None or lon_col is None):
            raise ValueError(
                "Dataset metadata does not contain recognized latitude/longitude columns."
            )
 
        # Détection de la colonne VS30 
        # On ne lève pas d'erreur si absente — on note None et on gérera
        # le fallback à 0.0 dans __getitem__.
        vs30_col = self.VS30_COL if self.VS30_COL in meta.columns else None
        if self.use_vs30 and vs30_col is None:
            print(
                "WARNING : use_vs30=True mais la colonne 'station_vs_30_mps' est "
                "absente dans ce dataset (probablement STEAD). "
                "Toutes les valeurs VS30 seront 0.0 (fallback silencieux)."
            )
 
        # Détection de la colonne instrument 
        # On essaie station_channels (INSTANCE) puis trace_channel (STEAD).
        instrument_col = next(
            (c for c in self.CHANNEL_COLS if c in meta.columns), None
        )
        if self.use_instrument and instrument_col is None:
            print(
                "WARNING : use_instrument=True mais aucune colonne instrument "
                "('station_channels' ou 'trace_channel') trouvée. "
                "Tous les vecteurs instrument seront 'other' (one-hot index 5)."
            )
        
 
        # ── Filtrage : garder les traces avec P et magnitude valides 
        has_p   = meta[p_col].notna().values
        has_mag = meta[mag_col].notna().values
        valid_mask = has_p & has_mag
 
        if self.use_coords:
            has_lat = meta[lat_col].notna().values
            has_lon = meta[lon_col].notna().values
            valid_mask = valid_mask & has_lat & has_lon
            # Note : on NE filtre PAS sur VS30 non-NaN — le fallback 0.0
            # est préférable à la perte de données d'entraînement.
 
        self.sb_dataset.filter(valid_mask)
        print(
            f"Filtered dataset for magnitude prediction. Retained {len(self.sb_dataset)} traces."
        )
 
        # ── Stocker les noms de colonnes pour __getitem__ 
        self.p_col          = p_col
        self.mag_col        = mag_col
        self.lat_col        = lat_col
        self.lon_col        = lon_col
        self.vs30_col       = vs30_col        # peut être None
        self.instrument_col = instrument_col  # peut être None

    def __len__(self):
        return len(self.sb_dataset)

    def __getitem__(self, idx):
        trace, meta = self.sb_dataset.get_sample(idx)
 
        p_arrival = int(meta[self.p_col])
        magnitude = float(meta[self.mag_col])
 
        # ── Extraction de la fenêtre temporelle ────────────────
        start_idx = p_arrival
        end_idx   = start_idx + self.window_len
 
        if end_idx > trace.shape[1]:
            # Padding avec zéros si la trace est trop courte
            window = np.zeros((trace.shape[0], self.window_len), dtype=np.float32)
            valid_len = trace.shape[1] - start_idx
            if valid_len > 0:
                window[:, :valid_len] = trace[:, start_idx:]
        else:
            window = trace[:, start_idx:end_idx].copy()
 
        # Normalisation par l'amplitude de pic 
        max_val = np.abs(window).max(axis=1, keepdims=True)
        max_val[max_val == 0] = 1.0
        window = window / max_val
 
        # ── Construction du dict de sortie 
        out = {
            "waveform":  torch.tensor(window,      dtype=torch.float32),
            "magnitude": torch.tensor([magnitude], dtype=torch.float32),
        }
 
        if self.use_coords:
            lat = float(meta[self.lat_col])
            lon = float(meta[self.lon_col])
            out["coords"] = torch.tensor([lat, lon], dtype=torch.float32)
 
        # ── VS30 ────────────────────────────────────────────────
        if self.use_vs30:
            # Lecture de la valeur brute (None si colonne absente)
            raw_vs30 = meta.get(self.vs30_col, None) if self.vs30_col else None
 
            try:
                vs30_val = float(raw_vs30) if raw_vs30 is not None else float("nan")
            except (TypeError, ValueError):
                vs30_val = float("nan")
 
            # log10 avec fallback 0.0 si invalide
            if np.isnan(vs30_val) or vs30_val <= 0.0:
                log_vs30 = 0.0
            else:
                log_vs30 = float(np.log10(vs30_val))
 
            out["vs30"] = torch.tensor([log_vs30], dtype=torch.float32)
 
        # ── TYPE D'INSTRUMENT ───────────────────────────────────
        if self.use_instrument:
            raw_channel = meta.get(self.instrument_col, None) if self.instrument_col else None
 
            code = "other"
            if raw_channel is not None:
                s = str(raw_channel).strip().upper()
                if len(s) >= 2:
                    prefix = s[:2]
                    code = prefix if prefix in INSTRUMENT_CODE_TO_IDX else "other"
 
            idx_code = INSTRUMENT_CODE_TO_IDX[code]
            one_hot  = np.zeros(N_INSTRUMENT_CLASSES, dtype=np.float32)
            one_hot[idx_code] = 1.0
            out["instrument"] = torch.tensor(one_hot, dtype=torch.float32)
 
        return out


def build_loaders_magnitude(
    dataset_name="INSTANCE",
    fraction=1.0,
    batch_size=128,
    window_len=200,
    use_coords=False,
    use_vs30=False,
    use_instrument=False,
):
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
            print(
                f"Subsampled {fraction*100:.1f}% of {split}: {num_samples} events selected."
            )

    common_kwargs = dict(
        window_len=window_len,
        use_coords=use_coords,
        use_vs30=use_vs30,
        use_instrument=use_instrument,
    )
    train_ds = MagnitudeDataset(train_sb, **common_kwargs)
    val_ds   = MagnitudeDataset(val_sb,   **common_kwargs)
    test_ds  = MagnitudeDataset(test_sb,  **common_kwargs)
 
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,  batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )

    return train_loader, val_loader, test_loader
