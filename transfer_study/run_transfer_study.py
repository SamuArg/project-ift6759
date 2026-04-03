"""
transfer_study/run_transfer_study.py
═══════════════════════════════════════════════════════════════════════════════
Script principal de l'étude de transfer learning STEAD → INSTANCE.

À lancer depuis la RACINE du projet :
    python transfer_study/run_transfer_study.py

Ce que fait ce script (dans l'ordre) :
────────────────────────────────────────
  1. ZERO-SHOT A  : base_lstm (poids STEAD) testé directement sur INSTANCE test
  2. ZERO-SHOT B  : EQTransformer (from_pretrained="stead") testé sur INSTANCE test
  3. FINE-TUNING  : Pour chaque (fraction, stratégie de gel) :
       fractions  = [1%, 5%, 10%, 15%, 25%] de INSTANCE train
       stratégies = ["partial" (LSTM+têtes), "full" (tout le réseau)]
       → Charge des poids STEAD frais
       → Applique le gel
       → Entraîne sur fraction% d'INSTANCE
       → Évalue sur INSTANCE test complet
  4. GOLD STANDARD : PhaseNet (from_pretrained="instance") testé sur INSTANCE test
  5. SAVE : Toutes les métriques sauvegardées dans un JSON unique

Structure de fichiers attendue :
─────────────────────────────────
  project_root/
  ├── dataset/load_dataset.py
  ├── models/base_lstm.py
  ├── training/training_loop.py
  ├── training/train.py
  ├── analysis/eval_functions.py
  ├── analysis/run_evaluation.py
  ├── trained_models/
  │   └── best_base_lstm_stead_h128_nocoords.pth   ← votre checkpoint
  └── transfer_study/
      ├── finetune_utils.py
      ├── run_transfer_study.py                     ← CE FICHIER
      └── analysis_plots.py

Résultats produits :
─────────────────────
  transfer_study/results/transfer_results.json      ← métriques complètes
  transfer_study/results/models/                    ← checkpoints fine-tunés
  transfer_study/results/logs/                      ← JSON de loss par epoch
  transfer_study/results/figures/                   ← courbes d'apprentissage
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import os
import time
import copy

import numpy as np
import torch

# ── Racine du projet sur le path ────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Code existant (non modifié) ─────────────────────────────────────────────
from training.train import build_model, build_loaders
from training.training_loop import train
from analysis.run_evaluation import run_evaluation

# ── Nos nouveaux utilitaires ────────────────────────────────────────────────
from transfer_study.finetune_utils import (
    load_stead_model,
    apply_freeze_strategy,
    build_finetune_optimizer,
    build_test_loader_only,
    save_results,
    make_serializable,
)


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — SEULE ZONE À MODIFIER
# ═══════════════════════════════════════════════════════════════════════════

# Chemin vers le checkpoint entraîné sur STEAD
STEAD_CHECKPOINT = "trained_models/best_base_lstm_stead_h128_nocoords.pth"
LSTM_HIDDEN      = 128   # doit correspondre à celui utilisé à l'entraînement

# Fractions du jeu d'entraînement INSTANCE à utiliser pour le fine-tuning
# 1% ≈ ~2 500 traces | 5% ≈ 12 500 | 10% ≈ 25 000 | 15% ≈ 37 500 | 25% ≈ 62 500
# (après filtre max_distance=100 km sur INSTANCE)
FRACTIONS = [0.01, 0.05, 0.10, 0.15, 0.25]

# Stratégies de gel et leur learning rate associé
# "partial" → LR élevé car seulement LSTM+têtes sont entraînés
# "full"    → LR bas pour éviter le catastrophic forgetting du CNN STEAD
STRATEGIES = {
    "partial": 1e-3,
    "full":    1e-4,
}

# Hyper-paramètres d'entraînement (fine-tuning)
N_EPOCHS     = 10     # épochs de fine-tuning (identique pour toutes les fractions)
BATCH_SIZE   = 64
MAX_DISTANCE = 100    # km — même filtre que lors de l'entraînement STEAD
SIGMA        = 10     # sigma gaussien pour les labels probabilistes
TYPE_LABEL   = "gaussian"

# Dossiers de sortie
RESULTS_JSON = "transfer_study/results/transfer_results.json"
MODELDIR     = "transfer_study/results/models"
LOGDIR       = "transfer_study/results/logs"
FIGDIR       = "transfer_study/results/figures"
SEED         = 42

# ═══════════════════════════════════════════════════════════════════════════


def set_seed(seed: int) -> None:
    """Fixe les seeds pour la reproductibilité."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _keep_scalar_metrics(raw: dict) -> dict:
    """
    Filtre la sortie brute pour ne garder que les métriques scalaires,
    et calcule les statistiques de résidus avant de jeter les distributions.
    """
    # 1. Calcul des statistiques de résidus (biais et variance temporelle)
    # Note : Si 'raw' ne contient que des erreurs absolues, il faudra s'assurer
    # que run_evaluation() renvoie bien (t_pred - t_true) pour capturer le biais directionnel.
    for phase in ["p", "s"]:
        err_key = f"errors_{phase}_wave" # Ajuster le nom selon ce que retourne run_evaluation
        if err_key in raw and len(raw[err_key]) > 0:
            raw[f"residual_mean_{phase}"] = np.mean(raw[err_key])
            raw[f"residual_std_{phase}"] = np.std(raw[err_key])
            
    # 2. Calcul explicite du Taux de Faux Positifs (FPR) sur le bruit
    if raw.get("n_noise_p", 0) > 0:
        raw["fpr_noise_p"] = raw.get("n_noise_fp_p", 0) / raw.get("n_noise_p")
    if raw.get("n_noise_s", 0) > 0:
        raw["fpr_noise_s"] = raw.get("n_noise_fp_s", 0) / raw.get("n_noise_s")

    KEEP = {
        "mae_p_wave",        "mae_s_wave",
        "f1_p_wave",         "f1_s_wave",
        "precision_p_wave",  "precision_s_wave",
        "recall_p_wave",     "recall_s_wave",
        "n_earthquake_p",    "n_earthquake_s",
        "n_noise_p",         "n_noise_s",
        "n_noise_fp_p",      "n_noise_fp_s",
        "total_fp_p",        "total_fp_s",
        "total_fn_p",        "total_fn_s",
        "residual_mean_p",   "residual_mean_s",
        "residual_std_p",    "residual_std_s",
        "fpr_noise_p",       "fpr_noise_s"
    }
    return {k: v for k, v in raw.items() if k in KEEP}


def evaluate_model(model, test_loader, device, label: str) -> dict:
    """
    Wrapper autour de run_evaluation() avec un en-tête clair.

    run_evaluation() gère déjà :
    - Le déplacement sur device
    - La conversion logits → probas (si nécessaire)
    - Les tuples (p, s) et (det, p, s) des différents modèles
    - Les métriques F1, MAE, Précision, Rappel
    """
    print(f"\n{'─'*65}")
    print(f"  ÉVALUATION : {label}")
    print(f"{'─'*65}")
    raw = run_evaluation(
        model=model,
        test_loader=test_loader,
        sampling_rate=100,           # Hz — identique pour STEAD et INSTANCE
        confidence_threshold=0.3,    # prob min pour considérer un pick
        noise_threshold=0.1,         # prob max du label pour considérer du bruit
        tolerance=0.1,               # ±0.1 s pour TP/FP classification
        device=device,
    )
    return _keep_scalar_metrics(raw)


# ═══════════════════════════════════════════════════════════════════════════
#  ÉTAPE 1 & 2 : ÉVALUATIONS ZERO-SHOT
# ═══════════════════════════════════════════════════════════════════════════

def run_zero_shot_evaluations(
    device,
    instance_test_loader_eqt,
    instance_test_loader_pn,
) -> dict:
    """
    Deux références zero-shot sur le test set d'INSTANCE :

    A. base_lstm avec les poids STEAD (notre modèle, aucune adaptation)
       → Montre le "coût de généralisation" de notre architecture sur INSTANCE.

    B. EQTransformer avec from_pretrained("stead") (état de l'art officiel)
       → Référence de comparaison : comment se comporte le meilleur modèle
         public sur INSTANCE sans adaptation ?

    Les deux utilisent le pipeline "eqtransformer" (fenêtre 6000 échantillons)
    qui est celui avec lequel base_lstm et EQT ont été entraînés sur STEAD.

    Note sur le pipeline PhaseNet (instance_test_loader_pn) :
    Ce loader (3001 échantillons, demean) est passé en argument mais n'est
    PAS utilisé ici — il est réservé à l'évaluation du gold standard PhaseNet.
    On le passe ici pour rendre la signature symétrique avec main().
    """
    zero_shot = {}

    # ── A : base_lstm zero-shot ───────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  ZERO-SHOT A : base_lstm (poids STEAD) → test INSTANCE")
    print("═" * 65)
    model_base = load_stead_model(STEAD_CHECKPOINT, lstm_hidden=LSTM_HIDDEN)
    # strategy="none" : tous les paramètres gelés (juste pour documenter l'intention)
    apply_freeze_strategy(model_base, "none")
    zero_shot["base_lstm_stead"] = evaluate_model(
        model_base,
        instance_test_loader_eqt,
        device,
        "base_lstm (STEAD) — zero-shot sur INSTANCE",
    )
    del model_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── B : EQTransformer zero-shot ───────────────────────────────────────
    # build_model("eqtransformer", checkpoint="stead") détecte que "stead"
    # n'est ni un fichier ni un dossier → is_sb_name=True
    # → sbm.EQTransformer.from_pretrained("stead")
    print("\n" + "═" * 65)
    print("  ZERO-SHOT B : EQTransformer (from_pretrained='stead') → test INSTANCE")
    print("═" * 65)
    model_eqt, _ = build_model("eqtransformer", checkpoint="stead")
    model_eqt.eval()
    zero_shot["eqtransformer_stead"] = evaluate_model(
        model_eqt,
        instance_test_loader_eqt,
        device,
        "EQTransformer (STEAD pretrained) — zero-shot sur INSTANCE",
    )
    del model_eqt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return zero_shot


# ═══════════════════════════════════════════════════════════════════════════
#  ÉTAPE 3 : EXPÉRIENCES DE FINE-TUNING
# ═══════════════════════════════════════════════════════════════════════════

def run_finetuning_experiments(device, instance_test_loader_eqt) -> dict:
    """
    Pour chaque combinaison (fraction, stratégie) :

    1. Charge une COPIE FRAÎCHE des poids STEAD
       → Important : chaque expérience repart des mêmes poids de départ.
         Sans ça, les expériences seraient dépendantes les unes des autres.

    2. Applique la stratégie de gel
       "partial" : gel stem+encoder+downsample, entraîne LSTM+têtes
       "full"    : tout dégelé (LR bas pour éviter catastrophic forgetting)

    3. Construit les loaders INSTANCE (train=fraction%, val=100%, test=100%)
       Le val et le test sont TOUJOURS complets pour évaluer correctement.
       build_loaders() de train.py n'applique dataset_fraction qu'au train.

    4. Fine-tune avec train() SANS le modifier
       On passe un optimizer pré-construit (que train() utilisera tel quel)
       et le learning_rate pour le scheduler OneCycleLR.

    5. Évalue sur le test INSTANCE COMPLET (instance_test_loader_eqt)
       Ce loader est construit une seule fois dans main() et réutilisé.
       Un DataLoader peut être itéré plusieurs fois sans problème.

    Retourne un dict imbriqué : results[strategy][str(fraction)] = métriques
    """
    finetune_results = {strategy: {} for strategy in STRATEGIES}

    for strategy, lr in STRATEGIES.items():
        for fraction in FRACTIONS:

            pct_label = f"{fraction * 100:.0f}%"
            run_label = f"finetune_{strategy}_f{int(fraction * 100):03d}pct"

            print(f"\n{'═' * 65}")
            print(f"  FINE-TUNING : stratégie={strategy.upper()}  fraction={pct_label} d'INSTANCE")
            print(f"{'═' * 65}")

            set_seed(SEED)  # reproductibilité par run

            # 1. Copie fraîche des poids STEAD
            model = load_stead_model(STEAD_CHECKPOINT, lstm_hidden=LSTM_HIDDEN)

            # 2. Gel des couches appropriées
            apply_freeze_strategy(model, strategy)

            # 3. Optimizer sur les seuls paramètres entraînables
            #    → évite de maintenir les états AdamW pour les params gelés
            optimizer = build_finetune_optimizer(model, strategy, lr)

            # 4. Loaders INSTANCE
            #    pipeline_type="eqtransformer" : fenêtre 6000 échantillons,
            #    normalisation detrend — cohérent avec comment base_lstm
            #    a été entraîné sur STEAD.
            print(f"\n  Chargement des données INSTANCE ({pct_label} pour train)…")
            train_loader, val_loader, test_loader = build_loaders(
                dataset="instance",
                pipeline_type="eqtransformer",  # fenêtre 6000, detrend norm
                fraction=fraction,              # SEULEMENT le train est sous-échantillonné
                batch_size=BATCH_SIZE,
                sigma=SIGMA,
                type_label=TYPE_LABEL,
                max_distance=MAX_DISTANCE,
                oversample=False,               # pas de sur-échantillonnage
                use_coords=False,
            )

            # 5. Fine-tuning avec la fonction train() EXISTANTE non modifiée
            #
            #    Points clés sur l'interaction avec train() :
            #    ─────────────────────────────────────────────
            #    a) On passe optimizer pré-construit → train() ne crée PAS
            #       un nouvel optimizer (il vérifie `if optimizer is None:`)
            #
            #    b) train() crée quand même un OneCycleLR sur cet optimizer
            #       avec max_lr=learning_rate (= lr ici). C'est correct :
            #       "partial" → max_lr=1e-3 | "full" → max_lr=1e-4
            #
            #    c) train() sauvegarde le meilleur checkpoint (val_loss)
            #       dans MODELDIR avec le nom run_label
            #
            #    d) train() évalue AUSSI sur test_loader en fin d'entraînement
            #       (évaluation rapide batch-F1 interne). Notre appel séparé
            #       à evaluate_model() ci-dessous donne les métriques complètes.
            #
            #    e) use_amp=True : Mixed Precision pour accélérer. Compatible
            #       avec base_lstm (contrairement à EQTransformer qui a des
            #       problèmes de précision float16 → use_amp=False pour EQT).
            print(f"\n  Entraînement ({N_EPOCHS} époques)…")
            model, _ = train(
                model=model,
                train_set=train_loader,
                validation_set=val_loader,
                test_set=test_loader,   # éval interne rapide à la fin
                model_name=run_label,
                device=device,
                optimizer=optimizer,    # pré-construit (uniquement params non-gelés)
                learning_rate=lr,       # utilisé par OneCycleLR comme max_lr
                epochs=N_EPOCHS,
                print_every=1,
                logdir=LOGDIR,
                figdir=FIGDIR,
                modeldir=MODELDIR,
                use_amp=True,
            )
            # Après train(), model a les poids du meilleur epoch (val_loss min)

            # 6. Évaluation complète sur le test INSTANCE complet
            #    (le même loader que pour le zero-shot → comparaison équitable)
            metrics = evaluate_model(
                model,
                instance_test_loader_eqt,
                device,
                f"base_lstm fine-tuné ({strategy}, {pct_label} INSTANCE)",
            )
            finetune_results[strategy][str(fraction)] = metrics

            # Nettoyage mémoire GPU entre chaque run
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return finetune_results


# ═══════════════════════════════════════════════════════════════════════════
#  ÉTAPE 4 : GOLD STANDARD — PhaseNet (from_pretrained="instance")
# ═══════════════════════════════════════════════════════════════════════════

def run_gold_standard(device, instance_test_loader_pn) -> dict:
    """
    Évalue PhaseNet pré-entraîné sur INSTANCE (poids officiels SeisBench).

    C'est le "plafond théorique" de performance sur la sismicité italienne :
    ce modèle a été entraîné sur la MAJORITÉ d'INSTANCE, donc il représente
    ce qu'on peut espérer au mieux sur ce jeu de données.

    Note importante sur le pipeline :
    PhaseNet utilise une fenêtre de 3001 échantillons et une normalisation
    demean (pas detrend). C'est pourquoi on lui passe instance_test_loader_pn
    (pipeline "phasenet") et non instance_test_loader_eqt. Évaluer PhaseNet
    avec le bon pipeline est essentiel pour avoir ses vraies performances.
    """
    print("\n" + "═" * 65)
    print("  GOLD STANDARD : PhaseNet (from_pretrained='instance') → test INSTANCE")
    print("═" * 65)
    # build_model("phasenet", checkpoint="instance") :
    # "instance" n'est ni fichier ni dossier → is_sb_name=True
    # → sbm.PhaseNet.from_pretrained("instance")
    model_pn, _ = build_model("phasenet", checkpoint="instance")
    model_pn.eval()
    metrics = evaluate_model(
        model_pn,
        instance_test_loader_pn,
        device,
        "PhaseNet (INSTANCE pretrained) — gold standard",
    )
    del model_pn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "═" * 65)
    print("  ÉTUDE DE TRANSFER LEARNING STEAD → INSTANCE")
    print("═" * 65)
    print(f"  Device      : {device}")
    print(f"  Checkpoint  : {STEAD_CHECKPOINT}")
    print(f"  Fractions   : {[f'{f*100:.0f}%' for f in FRACTIONS]}")
    print(f"  Stratégies  : {list(STRATEGIES.keys())}")
    print(f"  Époques     : {N_EPOCHS}")
    print(f"  Max dist.   : {MAX_DISTANCE} km")
    print("═" * 65)

    # Vérification du checkpoint
    if not os.path.isfile(STEAD_CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint introuvable : {STEAD_CHECKPOINT!r}\n"
            f"Placer le fichier .pth dans 'trained_models/' depuis la racine du projet."
        )

    t_start = time.time()

    # ─────────────────────────────────────────────────────────────────────
    # Construction des test loaders INSTANCE (UNE SEULE FOIS)
    # Ils sont réutilisés pour TOUTES les évaluations → comparaison équitable.
    #
    # Deux pipelines différents sont nécessaires :
    #   eqtransformer : 6000 échantillons, detrend → pour base_lstm et EQT
    #   phasenet      : 3001 échantillons, demean  → pour PhaseNet (gold std)
    #
    # Un DataLoader peut être itéré plusieurs fois : chaque for-loop sur
    # le loader repart du début. Pas besoin de le reconstruire à chaque fois.
    # ─────────────────────────────────────────────────────────────────────
    print("\n  Construction des test loaders INSTANCE (une seule fois)…")
    instance_test_eqt = build_test_loader_only(
        dataset_name="INSTANCE",
        pipeline_type="eqtransformer",   # 6000 échantillons, detrend
        batch_size=BATCH_SIZE,
        max_distance=MAX_DISTANCE,
        sigma=SIGMA,
        type_label=TYPE_LABEL,
        num_workers=4,
    )
    instance_test_pn = build_test_loader_only(
        dataset_name="INSTANCE",
        pipeline_type="phasenet",        # 3001 échantillons, demean
        batch_size=BATCH_SIZE,
        max_distance=MAX_DISTANCE,
        sigma=SIGMA,
        type_label=TYPE_LABEL,
        num_workers=4,
    )
    print("  ✓ Loaders INSTANCE prêts.")

    # ── Lancement des expériences ─────────────────────────────────────────
    all_results = {}

    # 1 & 2 : Évaluations zero-shot
    all_results["zero_shot"] = run_zero_shot_evaluations(
        device, instance_test_eqt, instance_test_pn
    )
    # Sauvegarde intermédiaire au cas où la suite planterait
    save_results(all_results, RESULTS_JSON)

    # 3 : Fine-tuning (la partie la plus longue)
    all_results["fine_tuning"] = run_finetuning_experiments(device, instance_test_eqt)
    save_results(all_results, RESULTS_JSON)

    # 4 : Gold standard
    all_results["gold_standard"] = {
        "phasenet_instance": run_gold_standard(device, instance_test_pn)
    }
    save_results(all_results, RESULTS_JSON)

    # ── Résumé final ──────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)

    print(f"\n{'═' * 65}")
    print(f"  Étude terminée en {h}h {m}m {s}s")
    print(f"  Résultats sauvegardés → {RESULTS_JSON}")
    print(f"  Pour l'analyse et les figures :")
    print(f"    python transfer_study/analysis_plots.py")
    print(f"{'═' * 65}")


if __name__ == "__main__":
    main()