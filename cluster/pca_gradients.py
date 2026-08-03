"""
Gradient plots: same pooled PCA embedding as clustering_analysis_with_cgm.py
(5 base vars + 20 curated CGM features, KMeans/Ward clustering on all of
them) -- but instead of coloring points by cluster label, color each point
by the RAW value of one variable at a time, using a continuous colormap.
This shows the direction each variable actually increases across the
cluster map, which a categorical cluster-color plot can't show directly.

7 variables get their own gradient panel: all 5 non-CGM base variables
(age, HbA1c, BMI, C-peptide, insulin) plus 2 CGM features chosen for having
the strongest combined PC1+PC2 loadings in the pooled with_cgm PCA
(tbr_lt70_pct and tir_70_180_pct -- also the two most standard/
interpretable CGM metrics clinically).

A companion biplot overlays all 7 variables' loading *directions* as
arrows on one panel (the classic PCA-biplot way to see every variable's
gradient direction at once, rather than one panel per variable).

This is a new, standalone script -- it does not modify or overwrite
clustering_analysis_with_cgm.py or its outputs, though it reuses the same
preprocessing so the PCA coordinates match exactly.

Outputs (in this directory):
  pca_gradients.png  -- 4x2 grid, one gradient-colored scatter per variable
  pca_biplot.png     -- single panel, KMeans-colored points + loading arrows
                        for all 7 variables
"""
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(OUT_DIR)
DATA_PATH = os.path.join(ROOT, "final_df.csv")
CGM_QC_MIN_PCT_ACTIVE = 70.0
BEST_K = 3  # matches clustering_analysis_with_cgm.py's chosen k

sys.path.insert(0, os.path.join(ROOT, "regression_no_correlated_features"))
from select_cgm_features_vif import CGM_FEATURES_VIF_PRUNED


def sanitize_column_name(col):
    return re.sub(r"[\[\]<]", "", col)


HBA1C_COL = sanitize_column_name("import_hba1c, Hemoglobin A1c/Hemoglobin.total in ")
CPEPTIDE_COL = sanitize_column_name("import_c_peptide, C peptide [Mass/volume] in Seru")
INSULIN_COL = sanitize_column_name("import_insulin, Insulin [Units/volume] in Serum o")
BMI_COL = "bmi_vsorres, BMI"

BASE_VAR_LABELS = {"age": "Age", HBA1C_COL: "HbA1c", BMI_COL: "BMI", CPEPTIDE_COL: "C-peptide", INSULIN_COL: "Insulin"}
BASE_FEATURE_COLS = ["age", HBA1C_COL, BMI_COL, CPEPTIDE_COL, INSULIN_COL]
FEATURE_COLS = BASE_FEATURE_COLS + CGM_FEATURES_VIF_PRUNED
LOG_COLS = {CPEPTIDE_COL, INSULIN_COL}

CGM_GRADIENT_VARS = ["tbr_lt70_pct", "tir_70_180_pct"]
GRADIENT_VARS = BASE_FEATURE_COLS + CGM_GRADIENT_VARS
GRADIENT_LABELS = {**BASE_VAR_LABELS, "tbr_lt70_pct": "TBR <70 mg/dL (%)", "tir_70_180_pct": "TIR 70-180 mg/dL (%)"}

df = pd.read_csv(DATA_PATH)
df.columns = [sanitize_column_name(c) for c in df.columns]
before = len(df)
df = df[df["qc_pct_active"] >= CGM_QC_MIN_PCT_ACTIVE]
df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
print(f"n={len(df)} (dropped {before - len(df)} for QC/missingness)")

X = df[FEATURE_COLS].copy()
for c in LOG_COLS:
    X[c] = np.log1p(X[c])
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=BEST_K, n_init=10, random_state=0).fit(X_scaled)
df["kmeans_cluster"] = kmeans.labels_

pca = PCA(n_components=2, random_state=0)
X_pca = pca.fit_transform(X_scaled)
df["pc1"], df["pc2"] = X_pca[:, 0], X_pca[:, 1]
var_explained = pca.explained_variance_ratio_
loadings = pd.DataFrame(pca.components_.T, index=FEATURE_COLS, columns=["PC1", "PC2"])
print(f"PCA variance explained: PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%}")

# --- Gradient plots: one panel per variable, colored by its raw value ---
fig, axes = plt.subplots(2, 4, figsize=(19, 9), facecolor="#fcfcfb")
axes = axes.flatten()

for ax, col in zip(axes, GRADIENT_VARS):
    ax.set_facecolor("#fcfcfb")
    sc = ax.scatter(df["pc1"], df["pc2"], c=df[col], cmap="viridis", s=10, alpha=0.7, edgecolor="none")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title(GRADIENT_LABELS[col], fontsize=11)
    ax.set_xlabel(f"PC1 ({var_explained[0]:.0%})", fontsize=8.5)
    ax.set_ylabel(f"PC2 ({var_explained[1]:.0%})", fontsize=8.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

for ax in axes[len(GRADIENT_VARS):]:
    ax.axis("off")

fig.suptitle("Gradient of each variable's raw value across the pooled PCA map\n(5 base vars + 20 curated CGM features, same embedding as clustering_analysis_with_cgm.py)", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pca_gradients.png"), dpi=200, facecolor=fig.get_facecolor())
plt.close(fig)
print("saved pca_gradients.png")

# --- Biplot: KMeans-colored points + loading-direction arrows for all 7 variables ---
# Use a robust (1st-99th percentile) plotting range rather than the raw min/max --
# a couple of extreme outlier points otherwise stretch the whole figure and force
# every arrow into a tiny unreadable cluster near the origin.
pc1_lo, pc1_hi = np.percentile(df["pc1"], [1, 99])
pc2_lo, pc2_hi = np.percentile(df["pc2"], [1, 99])
pad1, pad2 = 0.15 * (pc1_hi - pc1_lo), 0.15 * (pc2_hi - pc2_lo)
xlim = (pc1_lo - pad1, pc1_hi + pad1)
ylim = (pc2_lo - pad2, pc2_hi + pad2)

fig, ax = plt.subplots(figsize=(9, 8), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
CLUSTER_PALETTE = ["#2a78d6", "#eb6834", "#2ca858"]
for cl in range(BEST_K):
    sub = df[df["kmeans_cluster"] == cl]
    ax.scatter(sub["pc1"], sub["pc2"], s=8, alpha=0.25, color=CLUSTER_PALETTE[cl % len(CLUSTER_PALETTE)], zorder=2)

# Scale arrows to the robust plotting range (not the raw outlier-inflated range),
# so every arrow lands clearly inside the visible frame.
frame_half_extent = 0.75 * min(xlim[1] - xlim[0], ylim[1] - ylim[0]) / 2
max_loading = loadings.loc[GRADIENT_VARS, ["PC1", "PC2"]].abs().max().max()
arrow_scale = frame_half_extent / max_loading
for col in GRADIENT_VARS:
    x, y = loadings.loc[col, "PC1"] * arrow_scale, loadings.loc[col, "PC2"] * arrow_scale
    color = "#d64550" if col in CGM_GRADIENT_VARS else "#0b0b0b"
    ax.annotate("", xy=(x, y), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=color, linewidth=2.2), zorder=4)
    ax.text(x * 1.15, y * 1.15, GRADIENT_LABELS[col], fontsize=10, color=color,
            ha="center", va="center", fontweight="bold", zorder=5)

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel(f"PC1 ({var_explained[0]:.0%})", fontsize=11)
ax.set_ylabel(f"PC2 ({var_explained[1]:.0%})", fontsize=11)
ax.set_title(
    "Biplot: direction each variable increases in\n"
    "(black = base vars, red = CGM vars; points colored faintly by KMeans cluster)\n"
    "axes zoomed to 1st-99th percentile -- a few extreme outliers sit outside this frame",
    fontsize=12,
)
ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=1)
ax.axvline(0, color="#cccccc", linewidth=0.8, zorder=1)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pca_biplot.png"), dpi=200, facecolor=fig.get_facecolor())
plt.close(fig)
print("saved pca_biplot.png")
