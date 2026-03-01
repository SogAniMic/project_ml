"""
projet_ml.py — Pipeline ML complet : California Housing Prices
Michée SOGBOSSI | M1 IA | Ynov Toulouse 2025-2026

"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    BaggingRegressor, ExtraTreesRegressor,
    AdaBoostRegressor, StackingRegressor,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Config ──────────────────────────────────────────────────────────────────
from config import (
    DATA_PATH, FIGURES_DIR, TARGET_COL, TEST_SIZE, RANDOM_STATE,
    NUM_FEATURES, CAT_FEATURES,
    RF_N_ESTIMATORS, GB_N_ESTIMATORS, GB_LEARNING_RATE, GB_MAX_DEPTH,
    ADA_N_ESTIMATORS, PCA_VARIANCE_THRESHOLD, FIGURE_DPI,
)

# ── Helpers ──────────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  {name:<35} RMSE: {rmse:>10.2f}  MAE: {mae:>10.2f}  R²: {r2:.4f}")
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

def savefig(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Figure sauvegardée → figures/{name}")

# ── 1. Chargement ────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  1. CHARGEMENT DES DONNÉES")
print("="*65)

if not os.path.exists(DATA_PATH):
    sys.exit(f"\n❌ Fichier introuvable : {DATA_PATH}\n"
             f"   → Placez donnees2.csv dans le dossier data/")

df = pd.read_csv(DATA_PATH, sep=";")
print(f"\n  Shape      : {df.shape}")
print(f"  Colonnes   : {list(df.columns)}")
print(f"\n  Valeurs manquantes :\n{df.isnull().sum().to_string()}")

# ── 2. Visualisation géographique ────────────────────────────────────────────
print("\n" + "="*65)
print("  2. VISUALISATION")
print("="*65)

fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(df["longitude"], df["latitude"],
                c=df[TARGET_COL], cmap="jet", alpha=0.4,
                s=df["population"] / 100,
                vmin=df[TARGET_COL].min(), vmax=df[TARGET_COL].max())
plt.colorbar(sc, ax=ax, label="Valeur médiane ($)")
ax.set(xlabel="Longitude", ylabel="Latitude",
       title="Prix médians des logements en Californie\n(taille = population, couleur = prix)")
plt.tight_layout()
savefig("geo_visualisation.png")

# Matrice de corrélation
corr = df.select_dtypes(include=np.number).corr()
print(f"\n  Corrélations avec {TARGET_COL} :\n"
      f"{corr[TARGET_COL].sort_values(ascending=False).to_string()}")

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, ax=ax, square=True)
ax.set_title("Matrice de corrélation")
plt.tight_layout()
savefig("correlation_matrix.png")

# ── 3. Préparation ───────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  3. PRÉPARATION DES DONNÉES")
print("="*65)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print(f"\n  Train : {X_train.shape}  |  Test : {X_test.shape}")

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ]), NUM_FEATURES),
    ("cat", Pipeline([
        ("encoder", OrdinalEncoder()),
    ]), CAT_FEATURES),
])

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep  = preprocessor.transform(X_test)
print("  ✓ Pipeline de prétraitement appliqué")

# ── 4. Modèles ───────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  4. ENTRAÎNEMENT DES MODÈLES")
print("="*65)

results = {}

# --- Baseline ---
print("\n  [BASELINE]")
lr = LinearRegression().fit(X_train_prep, y_train)
results["Régression Linéaire (base)"] = evaluate(
    "Régression Linéaire (base)", y_test, lr.predict(X_test_prep))

# --- Bagging ---
print("\n  [BAGGING]")
rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train_prep, y_train)
results["Random Forest"] = evaluate("Random Forest", y_test, rf.predict(X_test_prep))

et = ExtraTreesRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
et.fit(X_train_prep, y_train)
results["Extra Trees"] = evaluate("Extra Trees", y_test, et.predict(X_test_prep))

bag = BaggingRegressor(estimator=DecisionTreeRegressor(),
                       n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
bag.fit(X_train_prep, y_train)
results["Bagging (Decision Tree)"] = evaluate("Bagging (Decision Tree)", y_test, bag.predict(X_test_prep))

# --- Boosting ---
print("\n  [BOOSTING]")
gb = GradientBoostingRegressor(n_estimators=GB_N_ESTIMATORS, learning_rate=GB_LEARNING_RATE,
                                max_depth=GB_MAX_DEPTH, random_state=RANDOM_STATE)
gb.fit(X_train_prep, y_train)
results["Gradient Boosting"] = evaluate("Gradient Boosting", y_test, gb.predict(X_test_prep))

ada = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=4),
                        n_estimators=ADA_N_ESTIMATORS, random_state=RANDOM_STATE)
ada.fit(X_train_prep, y_train)
results["AdaBoost"] = evaluate("AdaBoost", y_test, ada.predict(X_test_prep))

# --- Stacking ---
print("\n  [STACKING]")
stacking = StackingRegressor(
    estimators=[
        ("lr", LinearRegression()),
        ("dt", DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE)),
        ("rf", RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)),
    ],
    final_estimator=Ridge(),
    cv=5,
)
stacking.fit(X_train_prep, y_train)
results["Stacking (LR+DT+RF → Ridge)"] = evaluate(
    "Stacking (LR+DT+RF → Ridge)", y_test, stacking.predict(X_test_prep))

# ── 5. Comparaison ───────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  5. COMPARAISON DES MODÈLES")
print("="*65)

df_res = pd.DataFrame(results).T.sort_values("RMSE")
print(f"\n{df_res.round(2).to_string()}")

# Figure comparaison
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
colors = ["#e74c3c" if "base" in n else "#3498db" for n in df_res.index]
for metric, ax in zip(["RMSE", "MAE", "R2"], axes):
    bars = ax.barh(df_res.index, df_res[metric], color=colors)
    ax.set_title(metric, fontsize=13, fontweight="bold")
    for bar, val in zip(bars, df_res[metric]):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
axes[0].invert_xaxis()
axes[1].invert_xaxis()
plt.suptitle("Comparaison des modèles (rouge = baseline, bleu = ensemblistes)", fontsize=12)
plt.tight_layout()
savefig("model_comparison.png")

# Feature importance
feat_imp = pd.Series(rf.feature_importances_,
                     index=NUM_FEATURES + CAT_FEATURES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
feat_imp.plot(kind="barh", ax=ax, color="#2ecc71")
ax.set_title("Importance des variables – Random Forest", fontsize=12)
plt.tight_layout()
savefig("feature_importance.png")

# ── 6. Bonus PCA ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  6. BONUS — RÉDUCTION DIMENSIONNELLE (ACP)")
print("="*65)

pca_full = PCA().fit(X_train_prep)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_comp = int(np.argmax(cumvar >= PCA_VARIANCE_THRESHOLD) + 1)
print(f"\n  Composantes pour {PCA_VARIANCE_THRESHOLD*100:.0f}% de variance : {n_comp}/{X_train_prep.shape[1]}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
            pca_full.explained_variance_ratio_, color="#9b59b6", alpha=0.7)
axes[0].set(title="Variance expliquée par composante",
            xlabel="Composante", ylabel="Variance expliquée")
axes[1].plot(range(1, len(cumvar) + 1), cumvar * 100, "b-o", markersize=5)
axes[1].axhline(y=95, color="r", linestyle="--", label="95%")
axes[1].axvline(x=n_comp, color="g", linestyle="--", label=f"n={n_comp}")
axes[1].set(title="Variance cumulée", xlabel="Nb composantes", ylabel="Variance cumulée (%)")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.suptitle("Analyse en Composantes Principales (ACP)", fontsize=12)
plt.tight_layout()
savefig("pca_analysis.png")

pca = PCA(n_components=n_comp)
X_train_pca = pca.fit_transform(X_train_prep)
X_test_pca  = pca.transform(X_test_prep)

results_pca = {}
print(f"\n  Modèles avec PCA ({n_comp} composantes) :")
for name, model in [
    ("LR + PCA",               LinearRegression()),
    ("Random Forest + PCA",    RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)),
    ("Gradient Boosting + PCA",GradientBoostingRegressor(n_estimators=GB_N_ESTIMATORS, learning_rate=GB_LEARNING_RATE,
                                                          max_depth=GB_MAX_DEPTH, random_state=RANDOM_STATE)),
    ("Stacking + PCA",         StackingRegressor(estimators=[
                                    ("lr", LinearRegression()),
                                    ("dt", DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE)),
                                    ("rf", RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1))],
                                final_estimator=Ridge(), cv=5)),
]:
    model.fit(X_train_pca, y_train)
    results_pca[name] = evaluate(name, y_test, model.predict(X_test_pca))

# Figure comparaison avec/sans PCA
mapping = {
    "Régression Linéaire": ("Régression Linéaire (base)", "LR + PCA"),
    "Random Forest":       ("Random Forest",              "Random Forest + PCA"),
    "Gradient Boosting":   ("Gradient Boosting",          "Gradient Boosting + PCA"),
    "Stacking":            ("Stacking (LR+DT+RF → Ridge)","Stacking + PCA"),
}
comp = {k: {"RMSE sans PCA": results[v[0]]["RMSE"], "RMSE avec PCA": results_pca[v[1]]["RMSE"],
            "R² sans PCA":   results[v[0]]["R2"],   "R² avec PCA":   results_pca[v[1]]["R2"]}
        for k, v in mapping.items()}
df_comp = pd.DataFrame(comp).T
x, w = np.arange(len(df_comp)), 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, m1, m2, title in zip(axes,
                              ["RMSE sans PCA", "R² sans PCA"],
                              ["RMSE avec PCA", "R² avec PCA"],
                              ["RMSE (moins = mieux)", "R² (plus = mieux)"]):
    ax.bar(x - w/2, df_comp[m1], w, label="Sans PCA", color="#3498db", alpha=0.8)
    ax.bar(x + w/2, df_comp[m2], w, label="Avec PCA", color="#e67e22", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(df_comp.index, rotation=20, ha="right")
    ax.set_title(title, fontsize=12); ax.legend()
plt.suptitle("Impact de la PCA sur les performances", fontsize=12)
plt.tight_layout()
savefig("pca_comparison.png")

print("\n" + "="*65)
print("  ✅ PIPELINE TERMINÉ — Toutes les figures sont dans figures/")
print("="*65 + "\n")