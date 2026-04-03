import numpy as np
import seisbench.data as sbd
import seisbench.generate as sbg
from torch.utils.data import DataLoader, Subset


class AddCoords:
    def __init__(self, key="coords"):
        self.key = key

    def __call__(self, state_dict):
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

        lat = next((state_dict[c] for c in lat_cols if c in state_dict), float("nan"))
        lon = next((state_dict[c] for c in lon_cols if c in state_dict), float("nan"))

        if np.isnan(lat):
            lat = 0.0
        if np.isnan(lon):
            lon = 0.0

        # SeisBench state_dict arrays must be stored as (array, metadata_dict) tuples
        state_dict[self.key] = (np.array([lat, lon], dtype=np.float32), None)
        return state_dict


class SeisBenchPipelineWrapper:
    def __init__(
        self,
        dataset_name="STEAD",
        split="train",
        model_type="eqtransformer",
        component_order="ZNE",
        max_distance=None,
        transformation_shape="gaussian",
        transformation_sigma=10,
        dataset_fraction=1.0,
        oversample_magnitudes=False,
        use_coords=False,
        normalize=True,
    ):
        """
        A unified wrapper to generate training pipelines for various seismic models using SeisBench.

        Parameters:
        - dataset_name (str): "STEAD", "INSTANCE", "VCSEIS", "GEOFON", "TXED".
        - split (str): "train", "dev", or "test".
        - model_type (str): "eqtransformer", "phasenet", or "unet".
        - component_order (str): Channel order, usually "ZNE".
        - max_distance (float): Maximum distance in km for event selection (if supported by dataset).
        - transformation_shape (str): "gaussian" or "triangle" for label generation.
        - transformation_sigma (float): Sigma for probabilistic label generation.
        """
        self.dataset_name = dataset_name.upper()
        self.split = split.lower()
        self.model_type = model_type.lower()
        self.component_order = component_order
        self.max_distance = max_distance
        self.transformation_shape = transformation_shape
        self.transformation_sigma = transformation_sigma
        self.dataset_fraction = dataset_fraction
        self.oversample_magnitudes = oversample_magnitudes
        self.use_coords = use_coords
        self.normalize = normalize

        print(f"Loading {self.dataset_name} ({self.split} split)...")
        if self.dataset_name == "STEAD":
            dataset = sbd.STEAD(component_order=self.component_order)
        elif self.dataset_name == "INSTANCE":
            dataset = sbd.InstanceCountsCombined(component_order=self.component_order)
        elif self.dataset_name == "VCSEIS":
            dataset = sbd.VCSEIS(component_order=self.component_order)
        elif self.dataset_name == "GEOFON":
            dataset = sbd.GEOFON(component_order=self.component_order)
        elif self.dataset_name == "TXED":
            dataset = sbd.TXED(component_order=self.component_order)
        elif self.dataset_name == "DUMMY":
            dataset = sbd.DummyDataset(component_order=self.component_order)
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported.")

        if self.split == "train":
            self.dataset = dataset.train()
        elif self.split in ["dev", "val", "validation"]:
            self.dataset = dataset.dev()
        elif self.split == "test":
            self.dataset = dataset.test()

        # Pure noise traces (no P, no S) are kept for detection learning.
        # Only ambiguous traces (P present, S missing) are dropped.
        s_col_candidates = ["trace_s_arrival_sample", "trace_S_arrival_sample"]
        p_col_candidates = ["trace_p_arrival_sample", "trace_P_arrival_sample"]
        s_col = next(
            (c for c in s_col_candidates if c in self.dataset.metadata.columns), None
        )
        p_col_meta = next(
            (c for c in p_col_candidates if c in self.dataset.metadata.columns), None
        )
        if s_col is not None and p_col_meta is not None:
            has_s = self.dataset.metadata[s_col].notna().values
            has_p = self.dataset.metadata[p_col_meta].notna().values
            # Keep: has S  OR  is noise (no P and no S)
            keep = has_s | ~has_p
            n_before = len(self.dataset)
            self.dataset.filter(keep)
            n_dropped = n_before - keep.sum()
            print(
                f"Dropped {n_dropped} traces with P but no S annotation "
                f"({keep.sum()} remain, noise traces preserved)."
            )
        elif s_col is not None:
            # Fallback: no P column available, use old behaviour
            has_s = self.dataset.metadata[s_col].notna().values
            n_before = len(self.dataset)
            self.dataset.filter(has_s)
            print(
                f"Dropped {n_before - has_s.sum()} traces without S annotation "
                f"({has_s.sum()} remain)."
            )
        else:
            print("Warning: no S-arrival column found — skipping S-filter.")

        if self.max_distance is not None and self.dataset_name in ["STEAD", "INSTANCE"]:
            possible_cols = ["path_hyp_distance_km", "source_distance_km"]
            dist_col = next(
                (col for col in possible_cols if col in self.dataset.metadata.columns),
                None,
            )

            if dist_col:
                # Keep events within max_distance or with missing distance info (noise traces)
                mask = (
                    (self.dataset.metadata[dist_col] <= self.max_distance)
                    | (self.dataset.metadata[dist_col].isna())
                ).values
                self.dataset.filter(mask)
            print(
                f"Filtered dataset to max distance {self.max_distance} km: {mask.sum()} events remain."
            )

        if 0.0 < self.dataset_fraction < 1.0:
            total_events = len(self.dataset)
            num_samples = int(total_events * self.dataset_fraction)

            subsample_mask = np.zeros(total_events, dtype=bool)

            random_indices = np.random.choice(total_events, num_samples, replace=False)
            subsample_mask[random_indices] = True

            self.dataset.filter(subsample_mask)
            print(
                f"Subsampled {self.dataset_fraction*100:.1f}% of the dataset: {num_samples} events selected."
            )

        self._window_len = 6000 if self.model_type == "eqtransformer" else 3001

        if self.split == "train":
            self.generator = sbg.GenericGenerator(self.dataset)
        else:
            # Build control metadata: one row per trace, centred on P arrival
            self.generator = self._build_steered_generator()

        self._attach_pipeline()

        if self.split == "train" and self.oversample_magnitudes:
            balanced_indices = self._balance_magnitudes()
            if balanced_indices is not None:
                self.generator = Subset(self.generator, balanced_indices)

    def _balance_magnitudes(self):
        """
        Computes indices to oversample minority magnitude bins so all bins match
        the size of the largest bin. Noise traces (NaN magnitude) form their own bin.
        Returns a 1D numpy array of indices to pass to a torch Subset.
        """
        import pandas as pd

        meta = self.dataset.metadata
        n_original = len(meta)

        # 1. Identify magnitude column
        mag_cols = ["source_magnitude", "event_magnitude", "source_mag"]
        mag_col = next((c for c in mag_cols if c in meta.columns), None)

        if mag_col is None:
            print(
                "Warning: oversample_magnitudes=True but no magnitude column found. Skipping."
            )
            return None

        print(f"Oversampling magnitudes (column: '{mag_col}')...")
        mags = meta[mag_col].values

        # 2. Assign each trace to a bin ID
        # Bins: 0=Noise, 1=[0,1), 2=[1,2), 3=[2,3), 4=[3,4), 5=[4,5), 6=[5,6), 7=[6,∞)
        bin_edges = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        bin_ids = np.zeros(n_original, dtype=int)  # 0 is noise (where mag is NaN)

        valid_mask = ~pd.isna(mags)
        # Use np.digitize for the valid magnitudes.
        # Digitize returns indices 1..len(bins) based on where the value falls.
        # We shift by 1 so bin 1 is [0,1), bin 2 is [1,2), etc.
        bin_ids[valid_mask] = np.digitize(mags[valid_mask], bin_edges)

        # 3. Find the majority class size
        unique_bins, counts = np.unique(bin_ids, return_counts=True)
        max_count = counts.max()
        print(f"  Majority bin size: {max_count} traces")

        # 4. Oversample minority bins
        balanced_indices = []
        original_indices = np.arange(n_original)

        for b_id, b_count in zip(unique_bins, counts):
            idx_in_bin = original_indices[bin_ids == b_id]
            if b_count == max_count:
                balanced_indices.append(idx_in_bin)
            else:
                # Sample with replacement to match max_count
                sampled_idx = np.random.choice(idx_in_bin, size=max_count, replace=True)
                balanced_indices.append(sampled_idx)
                if b_id == 0:
                    label = "Noise"
                elif b_id == len(bin_edges):
                    label = f"M>={bin_edges[-1]}"
                else:
                    label = f"M[{bin_edges[b_id-1]},{bin_edges[b_id]})"
                print(f"  Oversampled '{label}': {b_count} -> {max_count}")

        # Flatten and shuffle
        balanced_indices = np.concatenate(balanced_indices)
        np.random.shuffle(balanced_indices)

        # 5. Return the shuffled oversampled indices
        print(
            f"Oversampling complete: index array grew from {n_original} to {len(balanced_indices)} traces."
        )
        return balanced_indices

    def _build_steered_generator(self):
        """Builds a SteeredGenerator for eval splits with control metadata."""
        import pandas as pd

        meta = self.dataset.metadata.copy()

        # Detect p-arrival column
        p_col = (
            "trace_p_arrival_sample"
            if "trace_p_arrival_sample" in meta.columns
            else "trace_P_arrival_sample"
        )

        # Build start/end sample centred on P arrival, fall back to trace start if NaN
        p_samples = meta[p_col].fillna(self._window_len // 2).astype(int)
        meta["start_sample"] = (p_samples - self._window_len // 2).clip(lower=0)
        meta["end_sample"] = meta["start_sample"] + self._window_len

        return sbg.SteeredGenerator(self.dataset, meta)

    def _attach_pipeline(self):
        """Configures the native SeisBench augmentations based on the target model."""
        augmentations = []

        window_len = self._window_len

        # --- 1. WAVEFORM PREPROCESSING & AUGMENTATION ---
        p_col = "trace_p_arrival_sample"
        s_col = "trace_s_arrival_sample"

        if p_col not in self.dataset.metadata.columns:
            p_col = "trace_P_arrival_sample"
            s_col = "trace_S_arrival_sample"

        if self.split == "train":
            if self.model_type == "eqtransformer":
                augmentations.extend(
                    [
                        sbg.OneOf(
                            [
                                sbg.WindowAroundSample(
                                    metadata_keys=[p_col, s_col],
                                    samples_before=6000,
                                    windowlen=12000,
                                    selection="random",
                                    strategy="variable",
                                ),
                                sbg.NullAugmentation(),
                            ],
                            probabilities=[2, 1],
                        ),
                        sbg.RandomWindow(windowlen=window_len, strategy="pad"),
                    ]
                )
            elif self.model_type == "phasenet":
                augmentations.extend(
                    [
                        sbg.OneOf(
                            [
                                sbg.WindowAroundSample(
                                    metadata_keys=[p_col, s_col],
                                    samples_before=3000,
                                    windowlen=6000,
                                    selection="random",
                                    strategy="variable",
                                ),
                                sbg.NullAugmentation(),
                            ],
                            probabilities=[2, 1],
                        ),
                        sbg.RandomWindow(windowlen=window_len, strategy="pad"),
                    ]
                )
            else:
                augmentations.extend(
                    [
                        sbg.OneOf(
                            [
                                sbg.WindowAroundSample(
                                    metadata_keys=[p_col, s_col],
                                    samples_before=window_len,
                                    windowlen=window_len * 2,
                                    selection="random",
                                    strategy="variable",
                                ),
                                sbg.NullAugmentation(),
                            ],
                            probabilities=[2, 1],
                        ),
                        sbg.RandomWindow(windowlen=window_len, strategy="pad"),
                    ]
                )
        else:
            augmentations.extend(
                [sbg.SteeredWindow(windowlen=window_len, strategy="pad")]
            )

        normalize_aug = []
        if self.normalize:
            norm_kwargs = {"amp_norm_axis": -1, "amp_norm_type": "peak"}
            if self.model_type == "eqtransformer":
                norm_kwargs["detrend_axis"] = -1
            else:
                norm_kwargs["demean_axis"] = -1
            normalize_aug = [sbg.Normalize(**norm_kwargs)]

        augmentations.extend(
            [
                sbg.ChangeDtype(np.float32),
                *normalize_aug,
                sbg.ProbabilisticLabeller(
                    label_columns=[p_col],
                    shape=self.transformation_shape,
                    sigma=self.transformation_sigma,
                    key=("X", "y_p"),
                    dim=0,
                ),
                sbg.ChangeDtype(np.float32, key="y_p"),
                sbg.ProbabilisticLabeller(
                    label_columns=[s_col],
                    shape=self.transformation_shape,
                    sigma=self.transformation_sigma,
                    key=("X", "y_s"),
                    dim=0,
                ),
                sbg.ChangeDtype(np.float32, key="y_s"),
                sbg.ChangeDtype(np.float32, key="X"),
            ]
        )

        if self.model_type == "eqtransformer":
            augmentations.extend(
                [
                    sbg.DetectionLabeller(
                        p_phases=[p_col], s_phases=[s_col], key=("X", "y_det")
                    ),
                    sbg.ChangeDtype(np.float32, key="y_det"),
                ]
            )

        if self.use_coords:
            augmentations.extend(
                [AddCoords(key="coords"), sbg.ChangeDtype(np.float32, key="coords")]
            )

        self.generator.add_augmentations(augmentations)

    def get_dataloader(self, batch_size=32, num_workers=4, shuffle=True):
        """Returns the PyTorch DataLoader."""
        return DataLoader(
            self.generator,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
