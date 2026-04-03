import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import seisbench.models as sbm

from models.base_lstm import SeismicPicker
from dataset.load_dataset import SeisBenchPipelineWrapper
from training.training_loop import train
from analysis.run_evaluation import run_evaluation
import numpy as np

# model : "base_lstm" | "phasenet" | "eqtransformer"
# dataset : "stead" | "instance"


configs = []
for dataset in ["instance", "stead"]:
    for use_coords in [False, True]:
        for lstm_hidden in [128, 64]:
            coords_str = "coords" if use_coords else "nocoords"
            configs.append(
                {
                    "model": "base_lstm",
                    "dataset": dataset,
                    "checkpoint": None,
                    "fraction": 1.0,
                    "n_epochs": 10,
                    "model_name": f"base_lstm_{dataset}_h{lstm_hidden}_{coords_str}_10epochs_phasenet_window",
                    "batch_size": 128,
                    "learning_rate": 1e-3,
                    "sigma": 10,
                    "type_label": "gaussian",
                    "max_distance": 100,
                    "lstm_hidden": lstm_hidden,
                    "lstm_layers": 2,
                    "dropout": 0.2,
                    "use_coords": use_coords,
                    "normalize": True,
                    "pipeline_type_override": "phasenet",
                }
            )

LOGDIR = "test_outputs/logs"
FIGDIR = "test_outputs/figures"
MODELDIR = "test_outputs/models"

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)

_SB_PRETRAINED = {
    "phasenet": ["stead", "instance", "geofon", "scedc", "ethz", "neic", "original"],
    "eqtransformer": [
        "stead",
        "instance",
        "geofon",
        "scedc",
        "ethz",
        "neic",
        "original",
    ],
}


def build_model(
    model_name: str,
    checkpoint: str = None,
    lstm_hidden: int = 128,
    dropout: float = 0.2,
    base_channels: int = 64,
    lstm_layers: int = 2,
    use_coords: bool = False,
) -> tuple[torch.nn.Module, str]:
    """
    Return (model, pipeline_model_type).
    """
    is_local_dir = checkpoint is not None and os.path.isdir(checkpoint)
    is_local_file = checkpoint is not None and os.path.isfile(checkpoint)
    is_sb_name = checkpoint is not None and not is_local_dir and not is_local_file

    if model_name == "base_lstm":
        model = SeismicPicker(
            in_channels=3,
            base_channels=base_channels,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
            use_coords=use_coords,
        )
        pipeline_type = "eqtransformer"  # 6000-sample window for this model

        if is_local_file:
            print(f"Loading base_lstm weights from {checkpoint} for fine-tuning…")
            model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True)
            )
        elif checkpoint is not None:
            raise ValueError(
                f"base_lstm only supports a local .pth checkpoint, got: {checkpoint!r}"
            )

    elif model_name == "phasenet":
        if is_local_dir:
            model = sbm.PhaseNet.load(checkpoint)
            print(f"Loaded PhaseNet (SeisBench dir) from {checkpoint}")
        elif is_sb_name:
            model = sbm.PhaseNet.from_pretrained(checkpoint)
            print(f"Loaded PhaseNet from_pretrained({checkpoint!r})")
        else:
            model = sbm.PhaseNet.from_pretrained("stead")
            if is_local_file:
                print(f"Loading PhaseNet weights from {checkpoint} for fine-tuning…")
                model.load_state_dict(
                    torch.load(checkpoint, map_location="cpu", weights_only=True)
                )
        pipeline_type = "phasenet"

    elif model_name == "eqtransformer":
        if is_local_dir:
            model = sbm.EQTransformer.load(checkpoint)
            print(f"Loaded EQTransformer (SeisBench dir) from {checkpoint}")
        elif is_sb_name:
            model = sbm.EQTransformer.from_pretrained(checkpoint)
            print(f"Loaded EQTransformer from_pretrained({checkpoint!r})")
        else:
            model = sbm.EQTransformer.from_pretrained("instance")
            if is_local_file:
                print(
                    f"Loading EQTransformer weights from {checkpoint} for fine-tuning…"
                )
                model.load_state_dict(
                    torch.load(checkpoint, map_location="cpu", weights_only=True)
                )
        pipeline_type = "eqtransformer"

    elif model_name == "unet":
        from models.Upgrade1_skip_connections import SeismicPickerUNet

        model = SeismicPickerUNet(
            in_channels=3,
            base_ch=32,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
        )
        pipeline_type = "phasenet"  # 3001 window, produces p, s

        if is_local_file:
            print(f"Loading unet weights from {checkpoint} for fine-tuning…")
            model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True)
            )
        elif checkpoint is not None:
            raise ValueError(
                f"unet only supports a local .pth checkpoint, got: {checkpoint!r}"
            )

    elif model_name == "unet_det":
        from models.Upgrade2_detection_head import SeismicPickerUNetDet

        model = SeismicPickerUNetDet(
            in_channels=3,
            base_ch=32,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
        )
        pipeline_type = "eqtransformer"  # 6000 window, produces det, p, s

        if is_local_file:
            print(f"Loading unet_det weights from {checkpoint} for fine-tuning…")
            model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True)
            )
        elif checkpoint is not None:
            raise ValueError(
                f"unet_det only supports a local .pth checkpoint, got: {checkpoint!r}"
            )

    elif model_name == "bilstm":
        from models.bilstm import SeismicBiLSTM

        model = SeismicBiLSTM(
            in_channels=3,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
        )
        pipeline_type = "eqtransformer"  # 6000 window

        if is_local_file:
            print(f"Loading bilstm weights from {checkpoint} for fine-tuning…")
            model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True)
            )
        elif checkpoint is not None:
            raise ValueError(
                f"bilstm only supports a local .pth checkpoint, got: {checkpoint!r}"
            )

    else:
        raise ValueError(
            f"Unknown model: {model_name!r}. "
            f"Choose 'base_lstm', 'bilstm', 'phasenet', 'eqtransformer', 'unet', or 'unet_det'."
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model : {model.__class__.__name__}  ({n_params:,} trainable params)")
    return model, pipeline_type


def build_loaders(
    dataset: str,
    pipeline_type: str,
    fraction: float,
    batch_size: int,
    sigma: int,
    type_label: str,
    max_distance: int,
    oversample: bool,
    use_coords: bool = False,
    normalize: bool = True,
):
    """Build train / val / test DataLoaders from SeisBenchPipelineWrapper."""
    common = dict(
        dataset_name=dataset,
        model_type=pipeline_type,
        component_order="ZNE",
        transformation_shape=type_label,
        transformation_sigma=sigma,
        max_distance=max_distance,
        normalize=normalize,
    )

    print(f"\nLoading {dataset.upper()} dataset (pipeline={pipeline_type})…")
    train_pipe = SeisBenchPipelineWrapper(
        split="train",
        dataset_fraction=fraction,
        oversample_magnitudes=oversample,
        use_coords=use_coords,
        **common,
    )
    val_pipe = SeisBenchPipelineWrapper(split="dev", use_coords=use_coords, **common)
    test_pipe = SeisBenchPipelineWrapper(split="test", use_coords=use_coords, **common)

    train_loader = train_pipe.get_dataloader(
        batch_size=batch_size, num_workers=16, shuffle=True
    )
    val_loader = val_pipe.get_dataloader(
        batch_size=batch_size, num_workers=16, shuffle=False
    )
    test_loader = test_pipe.get_dataloader(
        batch_size=batch_size, num_workers=16, shuffle=False
    )

    return train_loader, val_loader, test_loader


# Main


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, pipeline_type = build_model(
        config["model"],
        checkpoint=config["checkpoint"],
        lstm_hidden=config.get("lstm_hidden", 128),
        dropout=config.get("dropout", 0.2),
        base_channels=config.get("base_channels", 64),
        lstm_layers=config.get("lstm_layers", 2),
        use_coords=config.get("use_coords", False),
    )
    # Allow config to override the pipeline type (window size) independently of model
    loader_pipeline_type = config.get("pipeline_type_override", pipeline_type)
    train_loader, val_loader, test_loader = build_loaders(
        dataset=config["dataset"],
        pipeline_type=loader_pipeline_type,
        fraction=config["fraction"],
        batch_size=config["batch_size"],
        sigma=config["sigma"],
        type_label=config["type_label"],
        max_distance=config["max_distance"],
        oversample=config.get("oversample", False),
        use_coords=config.get("use_coords", False),
        normalize=config.get("normalize", True),
    )

    model, metrics = train(
        model=model,
        train_set=train_loader,
        validation_set=val_loader,
        test_set=test_loader,
        device=device,
        epochs=config["n_epochs"],
        learning_rate=config["learning_rate"],
        print_every=1,
        logdir=LOGDIR,
        figdir=FIGDIR,
        modeldir=MODELDIR,
        model_name=config["model_name"],
        use_amp=(config["model"] != "eqtransformer"),
    )
    print(
        f"\nRunning seismic pick evaluation on {config['dataset'].upper()} test split…"
    )

    results = run_evaluation(
        model=model,
        test_loader=test_loader,
        confidence_threshold=0.3,
        noise_threshold=0.1,
        tolerance=0.1,
        device=device,
    )


if __name__ == "__main__":

    for config in configs:
        print(f"\n{'='*40}\nRunning config: {config['model_name']}\n{'='*40}")
        main(config)
