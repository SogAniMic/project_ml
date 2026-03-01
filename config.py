# ==============================================================
# config.py — Configuration du projet ML - California Housing
# Michée SOGBOSSI | M1 IA | Ynov Toulouse 2025-2026
# ==============================================================

import os

# --- Chemins ---
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "donnees2.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
MODELS_DIR  = os.path.join(BASE_DIR, "models")

# Créer les dossiers s'ils n'existent pas
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

# --- Données ---
DATA_SEP       = ";"        # séparateur CSV
TARGET_COL     = "median_house_value"
TEST_SIZE      = 0.2
RANDOM_STATE   = 42

NUM_FEATURES = [
    "longitude", "latitude", "housing_median_age",
    "total_rooms", "total_bedrooms", "population",
    "households", "median_income"
]
CAT_FEATURES = ["ocean_proximity"]

# --- Modèles ensemblistes ---
RF_N_ESTIMATORS = 100
GB_N_ESTIMATORS = 200
GB_LEARNING_RATE = 0.1
GB_MAX_DEPTH     = 4
ADA_N_ESTIMATORS = 100

# --- PCA ---
PCA_VARIANCE_THRESHOLD = 0.95   # Conserver 95% de variance

# --- Affichage ---
FIGURE_DPI = 100