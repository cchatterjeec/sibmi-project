"""
Gradient + biplot views of the SAME PCA embedding as clustering_analysis_3d.py
/ pca_3d_clusters.png -- PCA fit on ONLY the 5 base variables (age, HbA1c,
BMI, C-peptide, insulin), full cohort, no CGM features in the fit.

(This replaces an earlier, wrong version of this script that re-fit PCA on
5 base vars + 20 CGM features -- a completely different embedding. Coordinates
here are read directly from pca_3d_cluster_assignments.csv, so they're
pixel-for-pixel identical to pca_3d_clusters.png.)

CGM features are added ONLY as color overlays / supplementary vectors,
never as inputs to the PCA fit:
  - Gradient panels: color the existing 5-base-var PCA points by two CGM
    features' raw values (tbr_lt70_pct, tir_70_180_pct), for the subset of
    participants who actually have CGM data (qc_pct_active >= 70%).
  - Biplot arrows: the 5 base variables use their REAL PCA loadings
    (pca_3d_loadings.csv). The 2 CGM features use "supplementary variable"
    vectors instead -- their correlation with PC1/PC2/PC3 among
    participants who have both -- since they were never part of the PCA
    fit and have no true loading. Drawn dashed/red and labeled as
    supplementary so they're not confused with real loadings.

Outputs (in this directory):
  pca_gradients_3d.png  -- 4x2 grid: 5 base-var gradients + 2 CGM gradients
  pca_biplot_3d.png     -- single 3D biplot: 5 real loadings (solid black) +
                           2 supplementary CGM vectors (dashed red)
"""
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(OUT_DIR)
CGM_QC_MIN_PCT_ACTIVE = 70.0
VIEW_ELEV, VIEW_AZIM = 20, 45
CGM_GRADIENT_VARS = ["tbr_lt70_pct", "tir_70_180_pct"]
CGM_LABELS = {"tbr_lt70_pct": "TBR <70 mg/dL (%)", "tir_70_180_pct": "TIR 70-180 mg/dL (%)"}
BASE_VARS = ["Age", "HbA1c", "BMI", "C-peptide", "Insulin"]


def sanitize_column_name(col):
    return re.sub(r"[\[\]<]", "", col)


HBA1C_COL = sanitize_column_name("import_hba1c, Hemoglobin A1c/Hemoglobin.total in ")
CPEPTIDE_COL = sanitize_column_name("import_c_peptide, C peptide [Mass/volume] in Seru")
INSULIN_COL = sanitize_column_name("import_insulin, Insulin [Units/volume] in Serum o")
BMI_COL = "bmi_vsorres, BMI"
RAW_COL = {"Age": "age", "HbA1c": HBA1C_COL, "BMI": BMI_COL, "C-peptide": CPEPTIDE_COL, "Insulin": INSULIN_COL}

# --- Load the EXACT 5-base-var PCA embedding already computed by clustering_analysis_3d.py ---
pca_df = pd.read_csv(os.path.join(OUT_DIR, "pca_3d_cluster_assignments.csv"))
loadings = pd.read_csv(os.path.join(OUT_DIR, "pca_3d_loadings.csv"), index_col=0)
print(f"Loaded existing 5-base-var PCA embedding: n={len(pca_df)}")

# --- Merge in raw base-var values (for gradient coloring) and the 2 CGM features ---
final_df = pd.read_csv(os.path.join(ROOT, "final_df.csv"))
final_df.columns = [sanitize_column_name(c) for c in final_df.columns]
keep_cols = ["participant_id", "qc_pct_active"] + [RAW_COL[v] for v in BASE_VARS] + CGM_GRADIENT_VARS
df = pca_df.merge(final_df[keep_cols], on="participant_id", how="left")

# --- Gradient plots ---
fig = plt.figure(figsize=(20, 10), facecolor="#fcfcfb")
panels = BASE_VARS + [CGM_LABELS[v] for v in CGM_GRADIENT_VARS]
axes = [fig.add_subplot(2, 4, i + 1, projection="3d") for i in range(len(panels))]

for ax, var in zip(axes[:5], BASE_VARS):
    ax.set_facecolor("#fcfcfb")
    col = RAW_COL[var]
    sc = ax.scatter(df["pc1"], df["pc2"], df["pc3"], c=df[col], cmap="viridis", s=8, alpha=0.6, edgecolor="none")
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1).ax.tick_params(labelsize=7)
    ax.set_title(var, fontsize=11)
    ax.set_xlabel("PC1", fontsize=7.5); ax.set_ylabel("PC2", fontsize=7.5); ax.set_zlabel("PC3", fontsize=7.5)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

for ax, col in zip(axes[5:], CGM_GRADIENT_VARS):
    ax.set_facecolor("#fcfcfb")
    sub = df.dropna(subset=[col])
    sub = sub[sub["qc_pct_active"] >= CGM_QC_MIN_PCT_ACTIVE]
    sc = ax.scatter(sub["pc1"], sub["pc2"], sub["pc3"], c=sub[col], cmap="viridis", s=8, alpha=0.6, edgecolor="none")
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1).ax.tick_params(labelsize=7)
    ax.set_title(f"{CGM_LABELS[col]}\n(supplementary -- not in PCA fit)", fontsize=10)
    ax.set_xlabel("PC1", fontsize=7.5); ax.set_ylabel("PC2", fontsize=7.5); ax.set_zlabel("PC3", fontsize=7.5)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    print(f"{CGM_LABELS[col]}: n={len(sub)} with CGM data (of {len(df)} total)")

fig.suptitle(
    "Gradient of each variable's raw value, on the SAME 5-base-variable PCA embedding as pca_3d_clusters.png\n"
    "(age, HbA1c, BMI, C-peptide, insulin only went into the PCA fit -- the 2 CGM panels are color overlays, not inputs)",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pca_gradients_3d.png"), dpi=200, facecolor=fig.get_facecolor())
plt.close(fig)
print("saved pca_gradients_3d.png")

# --- Biplot: real loadings (5 base vars) + supplementary correlation vectors (2 CGM vars) ---
pc1_lo, pc1_hi = np.percentile(df["pc1"], [1, 99])
pc2_lo, pc2_hi = np.percentile(df["pc2"], [1, 99])
pc3_lo, pc3_hi = np.percentile(df["pc3"], [1, 99])
xlim = (pc1_lo - 0.15 * (pc1_hi - pc1_lo), pc1_hi + 0.15 * (pc1_hi - pc1_lo))
ylim = (pc2_lo - 0.15 * (pc2_hi - pc2_lo), pc2_hi + 0.15 * (pc2_hi - pc2_lo))
zlim = (pc3_lo - 0.15 * (pc3_hi - pc3_lo), pc3_hi + 0.15 * (pc3_hi - pc3_lo))
frame_half_extent = 0.75 * min(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]) / 2

# Supplementary vectors for the 2 CGM features: correlation with PC1/PC2/PC3
# (the standard way to add a variable to a biplot post-hoc when it wasn't
# part of the PCA fit -- NOT a true loading).
supp_vectors = {}
for col in CGM_GRADIENT_VARS:
    sub = df.dropna(subset=[col])
    sub = sub[sub["qc_pct_active"] >= CGM_QC_MIN_PCT_ACTIVE]
    supp_vectors[col] = [sub["pc1"].corr(sub[col]), sub["pc2"].corr(sub[col]), sub["pc3"].corr(sub[col])]

all_vec_magnitudes = [loadings.loc[v].abs().max() for v in BASE_VARS] + [max(abs(x) for x in supp_vectors[c]) for c in CGM_GRADIENT_VARS]
arrow_scale = frame_half_extent / max(all_vec_magnitudes)

fig = plt.figure(figsize=(11, 10), facecolor="#fcfcfb")
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("#fcfcfb")
CLUSTER_PALETTE = ["#2a78d6", "#eb6834", "#2ca858"]
for cl in sorted(df["kmeans_cluster"].unique()):
    sub = df[df["kmeans_cluster"] == cl]
    ax.scatter(sub["pc1"], sub["pc2"], sub["pc3"], s=6, alpha=0.18, color=CLUSTER_PALETTE[int(cl) % len(CLUSTER_PALETTE)], zorder=2)

for var in BASE_VARS:
    x, y, z = (loadings.loc[var, pc] * arrow_scale for pc in ["PC1", "PC2", "PC3"])
    ax.quiver(0, 0, 0, x, y, z, color="#0b0b0b", linewidth=2.2, arrow_length_ratio=0.15, zorder=4)
    ax.text(x * 1.15, y * 1.15, z * 1.15, var, fontsize=9.5, color="#0b0b0b", ha="center", va="center", fontweight="bold", zorder=5)

for col in CGM_GRADIENT_VARS:
    x, y, z = (v * arrow_scale for v in supp_vectors[col])
    ax.quiver(0, 0, 0, x, y, z, color="#d64550", linewidth=2.2, linestyle="dashed", arrow_length_ratio=0.15, zorder=4)
    ax.text(x * 1.15, y * 1.15, z * 1.15, CGM_LABELS[col], fontsize=9.5, color="#d64550", ha="center", va="center", fontweight="bold", zorder=5)

ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
ax.set_xlabel("PC1", fontsize=11); ax.set_ylabel("PC2", fontsize=11); ax.set_zlabel("PC3", fontsize=11)
ax.set_title(
    "3D biplot on the 5-base-variable PCA embedding (same as pca_3d_clusters.png)\n"
    "solid black = real PCA loadings (age/HbA1c/BMI/C-peptide/insulin)\n"
    "dashed red = supplementary CGM vectors (correlation with PCs, not part of the fit)",
    fontsize=11.5,
)
ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pca_biplot_3d.png"), dpi=200, facecolor=fig.get_facecolor())
plt.close(fig)
print("saved pca_biplot_3d.png")
