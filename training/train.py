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
MODEL = "base_lstm"  # "base_lstm" | "phasenet" | "eqtransformer"
DATASET = "stead"  # "stead" | "instance" | "geofon" | "txed"
FRACTION = 0.1  # fraction of training data to use (1.0 = full)
N_EPOCHS = 10
MODEL_NAME = f"{MODEL}_{DATASET}_{N_EPOCHS}_{FRACTION}"
BATCH_SIZE = 128
LR = 1e-3
SIGMA = 10  # Gaussian label width (samples)
TYPE_LABEL = "gaussian"  # "gaussian" | "triangle"
MAX_DISTANCE = 100

LOGDIR = "test_outputs/logs"
FIGDIR = "test_outputs/figures"
MODELDIR = "test_outputs/models"
# ─────────────────────────────────────────────────────────────────────────────

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)


def build_model(model_name: str) -> tuple[torch.nn.Module, str]:
    """
    Return (model, pipeline_model_type) where pipeline_model_type controls
    which window length and augmentations the pipeline uses.
    """
    if model_name == "base_lstm":
        # SeismicPicker: CNN + BiLSTM, 6000-sample windows (same as EQT)
        model = SeismicPicker(
            in_channels=3,
            base_channels=64,
            lstm_hidden=32,
            lstm_layers=2,
            dropout=0.2,
        )
        pipeline_type = "eqtransformer"  # 6000-sample window for this model

    elif model_name == "phasenet":
        # Pretrained PhaseNet from SeisBench — fine-tune on new dataset
        model = sbm.PhaseNet.from_pretrained("stead")
        pipeline_type = "phasenet"  # 3001-sample window

    elif model_name == "eqtransformer":
        # Pretrained EQTransformer from SeisBench — fine-tune on new dataset
        model = sbm.EQTransformer.from_pretrained("instance")
        pipeline_type = "eqtransformer"  # 6000-sample window + y_det label

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
        split="train", dataset_fraction=fraction, **common
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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────
    model, pipeline_type = build_model(MODEL)

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_loaders(
        dataset=DATASET,
        pipeline_type=pipeline_type,
        fraction=FRACTION,
        batch_size=BATCH_SIZE,
        sigma=SIGMA,
        type_label=TYPE_LABEL,
        max_distance=MAX_DISTANCE,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    model, metrics = train(
        model=model,
        train_set=train_loader,
        validation_set=val_loader,
        test_set=test_loader,
        device=device,
        epochs=N_EPOCHS,
        learning_rate=LR,
        print_every=1,
        logdir=LOGDIR,
        figdir=FIGDIR,
        modeldir=MODELDIR,
        model_name=MODEL_NAME,
    )

    # ── Seismic evaluation (F1, MSE, Precision/Recall) ────────────────────
    print(f"\nRunning seismic pick evaluation on {DATASET.upper()} test split…")
    results = run_evaluation(
        model=model,
        test_loader=test_loader,
        confidence_threshold=0.3,
        noise_threshold=0.1,
        tolerance=0.1,  # ±0.1 s = ±10 samples at 100 Hz
        device=device,
    )
