import numpy as np
import seisbench.data as sbd
import seisbench.generate as sbg
from torch.utils.data import DataLoader


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
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported.")

        if self.split == "train":
            self.dataset = dataset.train()
        elif self.split in ["dev", "val", "validation"]:
            self.dataset = dataset.dev()
        elif self.split == "test":
            self.dataset = dataset.test()

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

        norm_kwargs = {"amp_norm_axis": -1, "amp_norm_type": "peak"}
        if self.model_type == "eqtransformer":
            norm_kwargs["detrend_axis"] = -1
        else:
            norm_kwargs["demean_axis"] = -1

        augmentations.extend(
            [
                sbg.ChangeDtype(np.float32),
                sbg.Normalize(**norm_kwargs),
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
