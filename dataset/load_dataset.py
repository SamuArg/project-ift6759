import numpy as np
import seisbench.data as sbd
import seisbench.generate as sbg
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class SeisBenchPipelineWrapper:
    def __init__(self, dataset_name="STEAD", split="train", model_type="eqtransformer", 
                 component_order="ZNE", apply_augmentations=False):
        """
        A unified wrapper to generate training pipelines for various seismic models using SeisBench.
        
        Parameters:
        - dataset_name (str): "STEAD", "INSTANCE", "VCSEIS", "GEOFON", "TXED".
        - split (str): "train", "dev", or "test".
        - model_type (str): "eqtransformer", "phasenet", or "unet".
        - component_order (str): Channel order, usually "ZNE".
        - apply_augmentations (bool): If True, applies stochastic transformations (noise, shift) during training.
        """
        self.dataset_name = dataset_name.upper()
        self.split = split.lower()
        self.model_type = model_type.lower()
        self.component_order = component_order
        self.apply_augmentations = apply_augmentations
        
        # 1. Load Dataset
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
            
        # 2. Apply Split
        if self.split == "train":
            self.dataset = dataset.train()
        elif self.split in ["dev", "val", "validation"]:
            self.dataset = dataset.dev()
        elif self.split == "test":
            self.dataset = dataset.test()
            
        # 3. Initialize SeisBench GenericGenerator
        self.generator = sbg.GenericGenerator(self.dataset)
        
        # 4. Attach Model-Specific Augmentations
        self._attach_pipeline()

    def _attach_pipeline(self):
        """Configures the native SeisBench augmentations based on the target model."""
        augmentations = []
        
        # EQTransformer expects 6000 samples, PhaseNet expects 3001
        window_len = 6000 if self.model_type == "eqtransformer" else 3001
        
        # --- 1. WAVEFORM PREPROCESSING & AUGMENTATION ---
        p_col = "trace_p_arrival_sample"
        s_col = "trace_s_arrival_sample"

        if p_col not in self.dataset.metadata.columns:
            p_col = "trace_P_arrival_sample"
            s_col = "trace_S_arrival_sample"
        
        if self.apply_augmentations and self.split == "train":
            print("Enabling stochastic augmentations (Train mode only)...")
            augmentations.extend([
                sbg.WindowAroundSample(
                    metadata_keys=[p_col, s_col], 
                    selection="random", 
                    samples_before=window_len // 2, 
                    strategy="pad", 
                    windowlen=window_len
                ),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="std")
            ])
            
            # Custom Callable Augmentation
            def add_stochastic_noise(state_dict):
                X_obj = state_dict["X"]
                is_tuple = isinstance(X_obj, tuple)
                X = X_obj[0] if is_tuple else X_obj

                if np.random.rand() < 0.5:
                    noise = np.random.normal(0, 0.1, X.shape)
                    X = X + noise
                    
                state_dict["X"] = (X, X_obj[1]) if is_tuple else X
            augmentations.append(add_stochastic_noise)
            
        else:
            print("Applying deterministic preprocessing...")
            augmentations.extend([
                sbg.RandomWindow(windowlen=window_len, strategy="pad"),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="std")
            ])
            
        # --- 2. FORMATTING FOR PYTORCH ---
        augmentations.append(sbg.ChangeDtype(np.float32, key="X"))
        
        
        if self.model_type == "eqtransformer":
            augmentations.extend([
                sbg.ProbabilisticLabeller(
                    label_columns=[p_col], 
                    shape="triangle", sigma=20, key=("X", "y_p"), dim=0
                ),
                sbg.ProbabilisticLabeller(
                    label_columns=[s_col], 
                    shape="triangle", sigma=20, key=("X", "y_s"), dim=0
                ),
                sbg.DetectionLabeller(
                    p_phases=[p_col], 
                    s_phases=[s_col], 
                    key=("X", "y_det")
                )
            ])
            
        elif self.model_type == "phasenet" or self.model_type == "unet":
            augmentations.extend([
                sbg.ProbabilisticLabeller(
                    label_columns=[p_col], 
                    shape="gaussian", sigma=10, key=("X", "y_p"), dim=0
                ),
                sbg.ProbabilisticLabeller(
                    label_columns=[s_col], 
                    shape="gaussian", sigma=10, key=("X", "y_s"), dim=0
                )
            ])
        else:
            raise ValueError(f"Model config for {self.model_type} not defined.")
            
        self.generator.add_augmentations(augmentations)

    def get_dataloader(self, batch_size=32, num_workers=4, shuffle=True):
        """Returns the PyTorch DataLoader."""
        return DataLoader(
            self.generator,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )