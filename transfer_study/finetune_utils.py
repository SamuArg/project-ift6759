"""
transfer_study/finetune_utils.py
═══════════════════════════════════════════════════════════════════════════════
Fonctions utilitaires pour l'étude de transfer learning STEAD → INSTANCE.
N'importe aucun fichier existant du projet ne doit être modifié.
Ces fonctions UTILISENT le code existant sans le toucher.
═══════════════════════════════════════════════════════════════════════════════

Rappel de l'architecture SeismicPicker (base_lstm.py) :
─────────────────────────────────────────────────────────
  stem        Conv1d(3 → 64, k=7)  +  BN  +  ReLU
  encoder     3× ResidualConvBlock (dilation 1, 2, 4)       ← feature extractor bas-niveau
  downsample  2× Conv1d strided  (64 → 128)                 ← compression temporelle
  lstm        BiLSTM(128-hidden × 2 layers)                 ← modélisation temporelle
  head_p/s    Conv1d(256 → 1, k=1)                          ← têtes de décision

Stratégies de gel
──────────────────
  "none"    → Tout gelé. Aucun gradient. Évaluation zero-shot pure.
  "partial" → stem + encoder + downsample gelés.
               Seuls le LSTM et les têtes (head_p, head_s) sont entraînés.
               Recommandé pour 1 – 5 % de données italiennes :
               peu de données → trop peu de signal pour contraindre tout le réseau.
  "full"    → Tout dégelé. Utiliser un LR plus faible (1e-4) pour éviter le
               catastrophic forgetting des features STEAD apprises sur CNN.
               Recommandé à partir de 10 – 25 % de données italiennes.
"""

import sys
import os
import json
import numpy as np
import torch
import torch.optim as optim

# ── Le dossier parent (racine du projet) doit être sur le sys.path ─────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.base_lstm import SeismicPicker
from dataset.load_dataset import SeisBenchPipelineWrapper

# ═══════════════════════════════════════════════════════════════════════════
#  1. CHARGEMENT DU MODÈLE STEAD
# ═══════════════════════════════════════════════════════════════════════════


def load_stead_model(checkpoint_path: str, lstm_hidden: int = 128) -> SeismicPicker:
    """
    Instancie un SeismicPicker avec les MÊMES hyper-paramètres que lors de
    l'entraînement sur STEAD (base_channels=64, lstm_layers=2, dropout=0.2,
    use_coords=False) et charge les poids sauvegardés.

    Retourne un modèle sur CPU. Appeler .to(device) avant entraînement/eval.

    Paramètres
    ──────────
    checkpoint_path : chemin vers le fichier .pth (ex: "trained_models/best_base_lstm_stead_h128_nocoords.pth")
    lstm_hidden     : doit correspondre à celui utilisé à l'entraînement (128 par défaut)
    """
    model = SeismicPicker(
        in_channels=3,
        base_channels=64,
        lstm_hidden=lstm_hidden,
        lstm_layers=2,
        dropout=0.2,
        use_coords=False,
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"  ✓ Poids STEAD chargés depuis {checkpoint_path!r}  ({n_params:,} paramètres au total)"
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  2. STRATÉGIES DE GEL
# ═══════════════════════════════════════════════════════════════════════════


def apply_freeze_strategy(model: SeismicPicker, strategy: str) -> None:
    """
    Gèle ou dégèle les couches de SeismicPicker pour le transfer learning.

    Modifie le modèle IN-PLACE (pas de valeur de retour).

    strategy : "none" | "partial" | "full"
    ──────────────────────────────────────
    "none"
        Tout est gelé (requires_grad=False sur tous les paramètres).
        À utiliser UNIQUEMENT pour l'évaluation zero-shot (pas d'entraînement).

    "partial"
        Gel : stem, encoder, downsample  (le CNN extracteur de features)
        Entraîne : lstm, head_p, head_s  (raisonnement temporel + décision)
        Justification : les features basses fréquences d'une impulsion sismique
        sont universelles (même physique des ondes partout dans le monde).
        Le biais géologique californien est surtout concentré dans la capacité
        du modèle à "décider" sur une fenêtre temporelle précise → c'est le LSTM
        et les têtes qu'il faut adapter. Avec peu de données (1–5%), c'est la
        seule stratégie qui ne mène pas à l'overfitting catastrophique.

    "full"
        Tout est dégelé (requires_grad=True partout).
        Le LR doit être bas (1e-4) pour ne pas "oublier" STEAD trop vite.
        À partir de 10–25% de données italiennes, assez de signal pour
        adapter les représentations intermédiaires à la géologie des Apennins.
    """
    if strategy == "none":
        for p in model.parameters():
            p.requires_grad = False

    elif strategy == "partial":
        # Étape 1 : tout geler
        for p in model.parameters():
            p.requires_grad = False
        # Étape 2 : dégeler les parties à adapter
        for module in [model.lstm, model.head_p, model.head_s]:
            for p in module.parameters():
                p.requires_grad = True
        # Note: model.lstm_dropout est nn.Dropout, pas de paramètres → sans effet

    elif strategy == "full":
        for p in model.parameters():
            p.requires_grad = True

    else:
        raise ValueError(
            f"Stratégie inconnue : {strategy!r}. Choisir parmi 'none', 'partial', 'full'."
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable
    pct = 100.0 * trainable / total if total > 0 else 0.0
    print(f"  Stratégie de gel : {strategy.upper()!r}")
    print(
        f"  Paramètres entraînables : {trainable:,} / {total:,}  ({pct:.1f}%)  "
        f"[gelés : {frozen:,}]"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  3. OPTIMIZER SPÉCIALISÉ POUR LE FINE-TUNING
# ═══════════════════════════════════════════════════════════════════════════


def build_finetune_optimizer(
    model: SeismicPicker,
    strategy: str,
    base_lr: float,
) -> optim.Optimizer:
    """
    Construit un AdamW qui ne couvre QUE les paramètres entraînables.

    Pourquoi ne pas juste utiliser model.parameters() ?
    ────────────────────────────────────────────────────
    Si on passe model.parameters() à AdamW avec strategy="partial", les
    paramètres gelés (requires_grad=False) n'auront pas de gradient, donc
    AdamW ne les mettra pas à jour. C'est fonctionnellement correct, MAIS
    AdamW maintient quand même des états (momentum, variance) pour ces
    paramètres, ce qui gaspille la mémoire GPU. En ne passant que les
    paramètres entraînables, on évite ce gaspillage.

    Paramètres
    ──────────
    model    : SeismicPicker avec freeze strategy déjà appliquée
    strategy : "partial" ou "full" (info pour le message)
    base_lr  : LR de base (1e-3 pour "partial", 1e-4 pour "full")
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError(
            "Aucun paramètre entraînable trouvé. "
            "Appeler apply_freeze_strategy() avant build_finetune_optimizer()."
        )
    optimizer = optim.AdamW(trainable_params, lr=base_lr, weight_decay=1e-4)
    print(
        f"  Optimiseur AdamW  lr={base_lr:.1e}  "
        f"({len(trainable_params)} tenseurs de paramètres)"
    )
    return optimizer


# ═══════════════════════════════════════════════════════════════════════════
#  4. CHARGEMENT D'UN TEST LOADER SEUL (sans train/val)
# ═══════════════════════════════════════════════════════════════════════════


def build_test_loader_only(
    dataset_name: str,
    pipeline_type: str,
    batch_size: int = 64,
    max_distance: int = 100,
    sigma: int = 10,
    type_label: str = "gaussian",
    num_workers: int = 4,
) -> torch.utils.data.DataLoader:
    """
    Construit UNIQUEMENT le DataLoader de test pour un dataset.

    Pourquoi ne pas utiliser build_loaders() de train.py ?
    ────────────────────────────────────────────────────────
    build_loaders() charge AUSSI les splits train et val, ce qui pour INSTANCE
    (~1.2M traces) prendrait beaucoup de temps alors qu'on n'en a pas besoin
    pour les évaluations zero-shot. Cette fonction ne charge que le split test.

    Paramètres
    ──────────
    pipeline_type : "eqtransformer" → fenêtre 6000 échantillons, detrend norm.
                    "phasenet"      → fenêtre 3001 échantillons, demean norm.
    max_distance  : même filtre que pendant l'entraînement STEAD (100 km)
    """
    pipe = SeisBenchPipelineWrapper(
        dataset_name=dataset_name,
        split="test",
        model_type=pipeline_type,
        component_order="ZNE",
        transformation_shape=type_label,
        transformation_sigma=sigma,
        max_distance=max_distance,
        use_coords=False,
    )
    return pipe.get_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  5. SÉRIALISATION JSON (conversion numpy → Python natif)
# ═══════════════════════════════════════════════════════════════════════════


def make_serializable(obj):
    """
    Convertit récursivement les types numpy en types Python natifs pour que
    json.dump() ne lève pas TypeError.

    run_evaluation() retourne des np.ndarray (abs_errors_*_wave) et des
    np.float64 — ce helper les convertit tous avant sauvegarde JSON.
    """
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    return obj


def save_results(results: dict, path: str) -> None:
    """
    Sauvegarde le dictionnaire complet de résultats en JSON.
    Crée les dossiers parents si nécessaire.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"Résultats sauvegardés: {path}")
