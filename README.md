# Phase Picking et Estimation de magnitude des séismes

## Table des matières

- [Phase Picking et Estimation de magnitude des séismes](#phase-picking-et-estimation-de-magnitude-des-séismes)
  - [Table des matières](#table-des-matières)
  - [Dépendances](#dépendances)
  - [Utilisation](#utilisation)

## Dépendances

Pour pouvoir utiliser les scripts, il faut installer les dépendances suivantes sous Python 3.12.3:

```bash
pip install -r requirements.txt
```

## Utilisation

Pour entrainer un modèle de phase picking, il suffit de modifier les configurations dans le fichier `training/train.py` et de lancer la commande suivante :

```bash
python training/train.py
```

Pour évaluer un modèle de phase picking, il suffit de modifier les configurations dans le fichier `analysis/evaluate_model.py` et de lancer la commande suivante :

```bash
python analysis/evaluate_model.py
```

Pour entrainer un modèle de prédiction de magnitude, il suffit de modifier les configurations dans le fichier `training/train_magnitude.py` et de lancer la commande suivante :

```bash
python training/train_magnitude.py
```

Pour évaluer un modèle de prédiction de magnitude, il suffit de modifier les configurations dans le fichier `analysis/evaluate_magnitude.py` et de lancer la commande suivante :

```bash
python analysis/evaluate_magnitude.py
```

Sinon, plusieurs notebooks sont disponibles dans le dossier `analysis/` pour explorer les données et les modèles.
