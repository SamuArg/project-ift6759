"""
transfer_study/analysis_plots.py
═══════════════════════════════════════════════════════════════════════════════
Script d'analyse et de visualisation pour l'étude de transfer learning.
Lit le JSON produit par run_transfer_study.py et génère :

  Figure 1  — Courbes F1 (onde P et S) en fonction du % de données INSTANCE
  Figure 2  — Courbes MAE en secondes en fonction du % de données INSTANCE
  Figure 3  — Comparaison en barres : zero-shot vs gold standard
  Console   — Tableau récapitulatif complet (toutes métriques, tous modèles)

À lancer depuis la RACINE du projet APRÈS run_transfer_study.py :
    python transfer_study/analysis_plots.py
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sans interface graphique (compatible serveur)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Configuration ─────────────────────────────────────────────────────────
RESULTS_JSON = "transfer_study/results/transfer_results.json"
FIGDIR       = "transfer_study/results/figures"

# Style général des figures
plt.rcParams.update({
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize":  9,
    "figure.dpi":      150,
})

# Couleurs constantes pour chaque "ligne" de modèle
COLORS = {
    "base_lstm_stead":       "tab:blue",
    "eqtransformer_stead":   "tab:green",
    "phasenet_instance":     "black",
    "partial":               "tab:orange",
    "full":                  "tab:red",
}
MARKERS = {"partial": "o", "full": "s"}
LINESTYLES = {
    "base_lstm_stead":     ":",
    "eqtransformer_stead": ":",
    "phasenet_instance":   "--",
}


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_results(path: str) -> dict:
    """Charge le JSON de résultats."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Fichier de résultats introuvable : {path!r}\n"
            f"Lancer d'abord : python transfer_study/run_transfer_study.py"
        )
    with open(path) as f:
        return json.load(f)


def m(d: dict, key: str, default=np.nan):
    """Lecture sécurisée d'une métrique (retourne NaN si absente)."""
    val = d.get(key, default)
    return float(val) if val is not None else np.nan


def get_finetune_curve(ft: dict, strategy: str, metric_key: str):
    """
    Extrait les vecteurs (x=fraction_pct, y=valeur_métrique) pour une stratégie.
    Les fractions sont stockées en string dans le JSON (ex: "0.01") → on convertit.
    """
    if strategy not in ft:
        return np.array([]), np.array([])
    items = sorted(ft[strategy].items(), key=lambda kv: float(kv[0]))
    xs = np.array([float(k) * 100 for k, _ in items])
    ys = np.array([m(v, metric_key)  for _, v in items])
    return xs, ys


def _save_fig(fig, name: str) -> None:
    """Sauvegarde une figure et ferme-la."""
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Figure sauvegardée → {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 1 : Courbes F1 vs % données INSTANCE
# ═══════════════════════════════════════════════════════════════════════════

def plot_f1_curves(results: dict) -> None:
    """
    Deux sous-graphes (P-wave / S-wave).

    Lignes horizontales pointillées = références fixes (pas de fine-tuning) :
      - base_lstm zero-shot    (bleu)
      - EQT zero-shot          (vert)
      - PhaseNet gold standard (noir)

    Courbes avec marqueurs = fine-tuning en cours :
      - partial (orange ronds)
      - full    (rouge carrés)

    Lecture du graphe :
    ───────────────────
    Si la courbe "full" croise la courbe "partial" entre 5% et 10%, cela
    signifie qu'à partir de ~10% de données italiennes il vaut mieux tout
    dégeler. C'est le seuil empirique de votre étude.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(
        "Transfer Learning STEAD→INSTANCE : F1-Score vs % données d'entraînement INSTANCE",
        fontweight="bold",
    )

    for ax, phase in zip(axes, ["p", "s"]):
        key = f"f1_{phase}_wave"

        # Références zero-shot (lignes horizontales)
        zs_base = m(results["zero_shot"]["base_lstm_stead"],      key)
        zs_eqt  = m(results["zero_shot"]["eqtransformer_stead"],  key)
        gs_pn   = m(results["gold_standard"]["phasenet_instance"], key)

        ax.axhline(
            zs_base, color=COLORS["base_lstm_stead"],
            linestyle=LINESTYLES["base_lstm_stead"], linewidth=1.8,
            label=f"base_lstm zero-shot ({zs_base:.3f})",
        )
        ax.axhline(
            zs_eqt, color=COLORS["eqtransformer_stead"],
            linestyle=LINESTYLES["eqtransformer_stead"], linewidth=1.8,
            label=f"EQTransformer zero-shot ({zs_eqt:.3f})",
        )
        ax.axhline(
            gs_pn, color=COLORS["phasenet_instance"],
            linestyle=LINESTYLES["phasenet_instance"], linewidth=2.2,
            label=f"PhaseNet gold standard ({gs_pn:.3f})",
        )

        # Courbes de fine-tuning
        for strategy in ["partial", "full"]:
            xs, ys = get_finetune_curve(
                results.get("fine_tuning", {}), strategy, key
            )
            if len(xs) == 0:
                continue
            ax.plot(
                xs, ys,
                color=COLORS[strategy],
                marker=MARKERS[strategy],
                linewidth=2, markersize=7,
                label=f"Fine-tune [{strategy}]",
            )

        ax.set_xlabel("% de données INSTANCE utilisées pour le fine-tuning")
        ax.set_ylabel("F1-Score")
        ax.set_title(f"Onde {'P' if phase == 'p' else 'S'}")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_ylim(bottom=0.0, top=1.05)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, "transfer_f1_curves.png")


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 2 : Courbes MAE vs % données INSTANCE
# ═══════════════════════════════════════════════════════════════════════════

def plot_mae_curves(results: dict) -> None:
    """
    Même structure que la figure F1, mais pour l'erreur temporelle (MAE).
    Les MAE sont en secondes. Plus la courbe descend, mieux le modèle
    localise temporellement les arrivées P et S.

    Interprétation :
    ────────────────
    Un modèle zero-shot peut avoir un bon F1 (il détecte les ondes) mais
    un MAE élevé (il les localise mal dans le temps). Ces deux courbes
    ensemble permettent de distinguer les deux aspects du problème.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(
        "Transfer Learning STEAD→INSTANCE : MAE (secondes) vs % données d'entraînement INSTANCE",
        fontweight="bold",
    )

    for ax, phase in zip(axes, ["p", "s"]):
        key = f"mae_{phase}_wave"

        zs_base = m(results["zero_shot"]["base_lstm_stead"],      key)
        zs_eqt  = m(results["zero_shot"]["eqtransformer_stead"],  key)
        gs_pn   = m(results["gold_standard"]["phasenet_instance"], key)

        ax.axhline(
            zs_base, color=COLORS["base_lstm_stead"],
            linestyle=LINESTYLES["base_lstm_stead"], linewidth=1.8,
            label=f"base_lstm zero-shot ({zs_base:.4f}s)",
        )
        ax.axhline(
            zs_eqt, color=COLORS["eqtransformer_stead"],
            linestyle=LINESTYLES["eqtransformer_stead"], linewidth=1.8,
            label=f"EQTransformer zero-shot ({zs_eqt:.4f}s)",
        )
        ax.axhline(
            gs_pn, color=COLORS["phasenet_instance"],
            linestyle=LINESTYLES["phasenet_instance"], linewidth=2.2,
            label=f"PhaseNet gold standard ({gs_pn:.4f}s)",
        )

        for strategy in ["partial", "full"]:
            xs, ys = get_finetune_curve(
                results.get("fine_tuning", {}), strategy, key
            )
            if len(xs) == 0:
                continue
            ax.plot(
                xs, ys,
                color=COLORS[strategy],
                marker=MARKERS[strategy],
                linewidth=2, markersize=7,
                label=f"Fine-tune [{strategy}]",
            )

        ax.set_xlabel("% de données INSTANCE utilisées pour le fine-tuning")
        ax.set_ylabel("MAE (secondes)")
        ax.set_title(f"Onde {'P' if phase == 'p' else 'S'}")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, "transfer_mae_curves.png")


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 3 : Comparaison zero-shot en barres
# ═══════════════════════════════════════════════════════════════════════════

def plot_zeroshot_comparison(results: dict) -> None:
    """
    Graphe en barres comparant les deux modèles zero-shot sur 4 métriques,
    avec le gold standard PhaseNet en ligne de référence.

    Ce graphe répond à : "Notre base_lstm zero-shot est-il meilleur ou moins
    bon qu'EQTransformer zero-shot sur les données italiennes ?"
    """
    model_keys   = ["base_lstm_stead", "eqtransformer_stead"]
    model_labels = ["base_lstm\n(zero-shot)", "EQTransformer\n(zero-shot)"]
    bar_colors   = [COLORS["base_lstm_stead"], COLORS["eqtransformer_stead"]]

    metrics_list = [
        ("f1_p_wave",  "F1 — Onde P"),
        ("f1_s_wave",  "F1 — Onde S"),
        ("mae_p_wave", "MAE Onde P (s)"),
        ("mae_s_wave", "MAE Onde S (s)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        "Comparaison des performances zero-shot sur le test set INSTANCE",
        fontweight="bold",
    )

    for ax, (metric_key, title) in zip(axes, metrics_list):
        values = [
            m(results["zero_shot"][k], metric_key) for k in model_keys
        ]
        gold = m(results["gold_standard"]["phasenet_instance"], metric_key)

        bars = ax.bar(
            model_labels, values,
            color=bar_colors, alpha=0.80, edgecolor="black", linewidth=0.8,
        )
        ax.axhline(
            gold, color="black", linestyle="--", linewidth=2.0,
            label=f"PhaseNet gold ({gold:.4f})",
        )

        # Annoter chaque barre avec sa valeur
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.003,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9,
                )

        ax.set_title(title)
        ax.set_ylim(bottom=0.0)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, "zeroshot_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 4 : Heatmap stratégie × fraction
# ═══════════════════════════════════════════════════════════════════════════

def plot_strategy_heatmap(results: dict) -> None:
    """
    Heatmap F1 moyen (P+S)/2 pour chaque cellule (stratégie × fraction).

    Permet de voir d'un coup d'œil :
    - Quelle stratégie est la meilleure pour chaque fraction ?
    - À quelle fraction atteint-on le plateau ?
    """
    ft = results.get("fine_tuning", {})
    strategies = [s for s in ["partial", "full"] if s in ft]
    if not strategies:
        print("  (pas de données de fine-tuning — heatmap ignorée)")
        return

    # Extraire les fractions communes
    all_fracs = sorted(
        set(float(k) for s in strategies for k in ft[s].keys())
    )
    frac_labels = [f"{f * 100:.0f}%" for f in all_fracs]

    data = np.full((len(strategies), len(all_fracs)), np.nan)
    for i, strategy in enumerate(strategies):
        for j, frac in enumerate(all_fracs):
            d = ft[strategy].get(str(frac), {})
            f1_p = m(d, "f1_p_wave")
            f1_s = m(d, "f1_s_wave")
            if not (np.isnan(f1_p) or np.isnan(f1_s)):
                data[i, j] = (f1_p + f1_s) / 2

    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, label="F1 moyen (P+S)/2")

    ax.set_xticks(range(len(frac_labels)))
    ax.set_xticklabels(frac_labels)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels([s.upper() for s in strategies])
    ax.set_xlabel("% données INSTANCE utilisées pour le fine-tuning")
    ax.set_ylabel("Stratégie de gel")
    ax.set_title(
        "F1 moyen (P+S)/2 par stratégie et fraction de données INSTANCE",
        fontweight="bold",
    )

    # Annoter les cellules
    for i in range(len(strategies)):
        for j in range(len(all_fracs)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(
                    j, i, f"{val:.3f}",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="black" if 0.3 < val < 0.85 else "white",
                )

    plt.tight_layout()
    _save_fig(fig, "strategy_fraction_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
#  Tableau récapitulatif (console)
# ═══════════════════════════════════════════════════════════════════════════

def print_summary_table(results: dict) -> None:
    """
    Affiche un tableau complet de toutes les métriques dans la console.

    Colonnes : F1-P | F1-S | MAE-P | MAE-S | Prec-P | Rec-P | Prec-S | Rec-S
    """
    COL = 42
    FMT_F = ">9.4f"
    FMT_S = ">10.4f"
    HDR = (
        f"  {'Modèle / Condition':<{COL}} "
        f"{'F1-P':>9} {'F1-S':>9} {'MAE-P(s)':>10} {'MAE-S(s)':>10} "
        f"{'Prec-P':>9} {'Rec-P':>9} {'Prec-S':>9} {'Rec-S':>9}"
    )
    SEP = "─" * (len(HDR) + 2)

    def _row(label, d):
        def _f(k): return f"{m(d, k):{FMT_F}}" if not np.isnan(m(d, k)) else f"{'N/A':>9}"
        def _fs(k): return f"{m(d, k):{FMT_S}}" if not np.isnan(m(d, k)) else f"{'N/A':>10}"
        print(
            f"  {label:<{COL}} "
            f"{_f('f1_p_wave')} {_f('f1_s_wave')} "
            f"{_fs('mae_p_wave')} {_fs('mae_s_wave')} "
            f"{_f('precision_p_wave')} {_f('recall_p_wave')} "
            f"{_f('precision_s_wave')} {_f('recall_s_wave')}"
        )

    print("\n" + "═" * (len(HDR) + 2))
    print("  TABLEAU RÉCAPITULATIF — TRANSFER LEARNING STEAD → INSTANCE")
    print("═" * (len(HDR) + 2))
    print(HDR)
    print(SEP)

    # Zero-shot
    _row("base_lstm (STEAD, zero-shot)",       results["zero_shot"]["base_lstm_stead"])
    _row("EQTransformer (STEAD, zero-shot)",   results["zero_shot"]["eqtransformer_stead"])
    print(SEP)

    # Fine-tuning
    for strategy in ["partial", "full"]:
        ft = results.get("fine_tuning", {}).get(strategy, {})
        if not ft:
            continue
        for frac_str, d in sorted(ft.items(), key=lambda kv: float(kv[0])):
            pct = float(frac_str) * 100
            _row(f"Fine-tune [{strategy}]  {pct:.0f}% INSTANCE", d)
        print(SEP)

    # Gold standard
    _row("PhaseNet (INSTANCE, gold standard)", results["gold_standard"]["phasenet_instance"])
    print("═" * (len(HDR) + 2))


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f"Chargement des résultats depuis {RESULTS_JSON!r}…")
    results = load_results(RESULTS_JSON)

    print_summary_table(results)

    print("\nGénération des figures…")
    plot_f1_curves(results)
    plot_mae_curves(results)
    plot_zeroshot_comparison(results)
    plot_strategy_heatmap(results)

    print(f"\n  Toutes les figures sont dans : {FIGDIR}/")
    print("  Fichiers produits :")
    for fname in [
        "transfer_f1_curves.png",
        "transfer_mae_curves.png",
        "zeroshot_comparison.png",
        "strategy_fraction_heatmap.png",
    ]:
        path = os.path.join(FIGDIR, fname)
        exists = "✓" if os.path.isfile(path) else "✗"
        print(f"    {exists} {path}")


if __name__ == "__main__":
    main()