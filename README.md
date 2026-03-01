# Projet ML — California Housing Prices

**Auteur** : Michée SOGBOSSI  
**Formation** : M1 Intelligence Artificielle — Ynov Toulouse  
**Année** : 2025 – 2026

---

## Objectif

Prédire le **prix médian des habitations** dans les districts de Californie à partir de données de recensement. Ce projet couvre un pipeline ML complet : exploration, préparation, modèles ensemblistes et réduction dimensionnelle (ACP).

---

## Structure du projet

```
ml_project/
│
├── data/
│   └── donnees2.csv          ← Jeu de données (à placer ici)
│
├── figures/                  ← Générées automatiquement à l'exécution
│   ├── geo_visualisation.png
│   ├── correlation_matrix.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   ├── pca_analysis.png
│   └── pca_comparison.png
│
├── config.py                 ← Tous les paramètres configurables
├── projet_ml.py              ← Script principal
├── notebook.ipynb            ← Version Jupyter Notebook
├── requirements.txt          ← Dépendances Python
└── README.md                 ← Ce fichier
```

---

## Installation

### 1. Cloner / récupérer le projet

Placez-vous dans le dossier du projet :
```bash
cd ml_project
```

### 2. (Recommandé) Créer un environnement virtuel

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Placer les données

Copiez `donnees2.csv` dans le dossier `data/` :
```bash
mkdir data
cp /chemin/vers/donnees2.csv data/
```

---

## Utilisation

### Option A — Script Python (terminal)

```bash
python projet_ml.py
```

Les figures sont sauvegardées dans `figures/`, les résultats s'affichent dans le terminal.

### Option B — Jupyter Notebook

```bash
jupyter notebook notebook.ipynb
```

Exécutez les cellules une par une avec `Shift + Entrée`, ou toutes d'un coup via **Kernel → Restart & Run All**.

### Option C — VS Code

Ouvrez `projet_ml.py` et cliquez sur ▶️ **Run Python File** (nécessite l'extension Python).

---

## Configuration

Tous les paramètres sont centralisés dans `config.py` :

| Paramètre | Valeur par défaut | Description |
|-----------|-------------------|-------------|
| `TEST_SIZE` | `0.2` | Part du jeu de test (20%) |
| `RANDOM_STATE` | `42` | Graine aléatoire (reproductibilité) |
| `RF_N_ESTIMATORS` | `100` | Nombre d'arbres (Random Forest) |
| `GB_N_ESTIMATORS` | `200` | Nombre d'itérations (Gradient Boosting) |
| `GB_LEARNING_RATE` | `0.1` | Taux d'apprentissage (Gradient Boosting) |
| `PCA_VARIANCE_THRESHOLD` | `0.95` | Seuil de variance conservée par PCA |
| `FIGURE_DPI` | `100` | Résolution des figures |

---

## Résultats

| Modèle | RMSE | R² |
|--------|------|----|
| Gradient Boosting ⭐ | 50 275 $ | 0.82 |
| Stacking (LR+DT+RF → Ridge) | 50 333 $ | 0.82 |
| Random Forest | 50 354 $ | 0.82 |
| Bagging (Decision Tree) | 50 364 $ | 0.82 |
| Extra Trees | 52 148 $ | 0.80 |
| **Régression Linéaire (baseline)** | 70 474 $ | 0.64 |
| AdaBoost | 96 480 $ | 0.33 |

L'apprentissage ensembliste réduit le RMSE de **~28%** par rapport à la baseline.  
La PCA (5 composantes → 95% variance) **dégrade** légèrement les performances dans ce cas.

---

## Dépendances principales

- Python ≥ 3.9
- scikit-learn ≥ 1.2
- pandas, numpy, matplotlib, seaborn
- jupyter (pour le notebook)