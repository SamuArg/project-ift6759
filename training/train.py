"""
Training entry point — real models, real data.

Trains a seismic phase-picking model using the SeisBenchPipelineWrapper
and evaluates it on the test split using run_evaluation().

Supported models:
  - "base_lstm"      : SeismicPicker (CNN + BiLSTM), trained from scratch
  - "phasenet"       : Pretrained SeisBench PhaseNet, fine-tuned
  - "eqtransformer"  : Pretrained SeisBench EQTransformer, fine-tuned

Run from project root:
    python training/test_training.py
"""

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

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ←  edit this block to switch model / dataset / hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

# model : "base_lstm" | "phasenet" | "eqtransformer"
# dataset : "stead" | "instance" | "geofon" | "txed"
# checkpoint : See build_model() for resolution order. Examples:
#   None                  → base_lstm: random init; phasenet/eqtransformer: default pretrained
#   "instance"            → phasenet/eqtransformer: from_pretrained("instance")
#   "stead"               → phasenet/eqtransformer: from_pretrained("stead")
#   "path/to/weights.pth" → any model: load_state_dict from local .pth file
#   "path/to/sb_dir/"     → phasenet/eqtransformer: SeisBench .load() from local dir


configs = [
        {
        "model": "base_lstm",
        "dataset": "instance",
        "checkpoint": None,
        "fraction": 1.0,
        "n_epochs": 10,
        "model_name": "base_lstm_instance_10_epochs_full_h128_d0.2_v2",
        "batch_size": 128,
        "learning_rate": 1e-3,
        "sigma": 10,
        "type_label": "gaussian",
        "max_distance": 100,
        "lstm_hidden": 128,
        "dropout": 0.2,
    },
    {
        "model": "base_lstm",
        "dataset": "instance",
        "checkpoint": None,
        "fraction": 1.0,
        "n_epochs": 10,
        "model_name": "base_lstm_instance_10_epochs_full_h64_d0_v2",
        "batch_size": 128,
        "learning_rate": 1e-3,
        "sigma": 10,
        "type_label": "gaussian",
        "max_distance": 100,
        "lstm_hidden": 64,
        "dropout": 0,
    },
    {
        "model": "base_lstm",
        "dataset": "instance",
        "checkpoint": None,
        "fraction": 1.0,
        "n_epochs": 10,
        "model_name": "base_lstm_instance_10_epochs_full_h32_d0_v2",
        "batch_size": 128,
        "learning_rate": 1e-3,
        "sigma": 10,
        "type_label": "gaussian",
        "max_distance": 100,
        "lstm_hidden": 32,
        "dropout": 0,
    },
    {
        "model": "base_lstm",
        "dataset": "instance",
        "checkpoint": None,
        "fraction": 1.0,
        "n_epochs": 10,
        "model_name": "base_lstm_instance_10_epochs_full_h32_d0.2_v2",
        "batch_size": 128,
        "learning_rate": 1e-3,
        "sigma": 10,
        "type_label": "gaussian",
        "max_distance": 100,
        "lstm_hidden": 32,
        "dropout": 0.2,
    },
    {
        "model": "base_lstm",
        "dataset": "instance",
        "checkpoint": None,
        "fraction": 1.0,
        "n_epochs": 10,
        "model_name": "base_lstm_instance_10_epochs_full_h64_d0.2_v2",
        "batch_size": 128,
        "learning_rate": 1e-3,
        "sigma": 10,
        "type_label": "gaussian",
        "max_distance": 100,
        "lstm_hidden": 64,
        "dropout": 0.2,
    },
    {
        "model": "base_lstm",
        "dataset": "instance",
        "checkpoint": None,
        "fraction": 1.0,
        "n_epochs": 10,
        "model_name": "base_lstm_instance_10_epochs_full_h128_d0_v2",
        "batch_size": 128,
        "learning_rate": 1e-3,
        "sigma": 10,
        "type_label": "gaussian",
        "max_distance": 100,
        "lstm_hidden": 128,
        "dropout": 0,
    },
]

configs = [{
    "model": "base_lstm",
    "dataset": "instance",
    "checkpoint": None,
    "fraction": 0.1,
    "n_epochs": 2,
    "model_name": "base_lstm_instance_2_epochs_0.1_h128_c128_l2_d0.2",
    "batch_size": 32,
    "learning_rate": 1e-3,
    "sigma": 10,
    "type_label": "gaussian",
    "max_distance": 100,
    "lstm_hidden": 128,
    "dropout": 0.2,
    "base_channels": 128,
    "lstm_layers": 2,
    "oversample": True,
}]

LOGDIR = "test_outputs/logs"
FIGDIR = "test_outputs/figures"
MODELDIR = "test_outputs/models"
# ─────────────────────────────────────────────────────────────────────────────

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)


# Known SeisBench pretrained weight keys per model class.
# Any string that is NOT an existing path is treated as a pretrained name.
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
) -> tuple[torch.nn.Module, str]:
    """
    Return (model, pipeline_model_type) where pipeline_model_type controls
    which window length and augmentations the pipeline uses.

    CHECKPOINT resolution order (applies to phasenet / eqtransformer):
      1. None              → from_pretrained(default)
      2. existing dir path → SeisBench .load(checkpoint)
      3. existing .pth     → from_pretrained(default) then load_state_dict(.pth)
      4. plain string      → from_pretrained(checkpoint)  ← NEW: e.g. "instance"
    """
    is_local_dir = checkpoint is not None and os.path.isdir(checkpoint)
    is_local_file = checkpoint is not None and os.path.isfile(checkpoint)
    # A plain name like "instance" or "stead" that is NOT an existing path
    is_sb_name = checkpoint is not None and not is_local_dir and not is_local_file

    if model_name == "base_lstm":
        model = SeismicPicker(
            in_channels=3,
            base_channels=base_channels,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
        )
        pipeline_type = "eqtransformer"  # 6000-sample window for this model

        # For base_lstm checkpoint must be a local .pth (no SeisBench hub for it)
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
            # None → default pretrained; local .pth handled below
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

    else:
        raise ValueError(
            f"Unknown model: {model_name!r}. "
            f"Choose 'base_lstm', 'phasenet', or 'eqtransformer'."
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
):
    """Build train / val / test DataLoaders from SeisBenchPipelineWrapper."""
    common = dict(
        dataset_name=dataset,
        model_type=pipeline_type,
        component_order="ZNE",
        transformation_shape=type_label,
        transformation_sigma=sigma,
        max_distance=max_distance,
    )

    print(f"\nLoading {dataset.upper()} dataset (pipeline={pipeline_type})…")
    train_pipe = SeisBenchPipelineWrapper(
        split="train", dataset_fraction=fraction, oversample_magnitudes=oversample, **common
    )
    val_pipe = SeisBenchPipelineWrapper(split="dev", **common)
    test_pipe = SeisBenchPipelineWrapper(split="test", **common)

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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


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
    )
    train_loader, val_loader, test_loader = build_loaders(
        dataset=config["dataset"],
        pipeline_type=pipeline_type,
        fraction=config["fraction"],
        batch_size=config["batch_size"],
        sigma=config["sigma"],
        type_label=config["type_label"],
        max_distance=config["max_distance"],
        oversample=config.get("oversample", False),
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
        tolerance=0.1,  # ±0.1 s = ±10 samples at 100 Hz
        device=device,
    )


if __name__ == "__main__":

    for config in configs:
        print(f"\n{'='*40}\nRunning config: {config['model_name']}\n{'='*40}")
        main(config)
