"""
Stage 1.5: Canonical Correlation Analysis (CCA) between Set A (the 5 base
clinical variables that define the clusters) and Set B (the 20 curated
CGM features), on participants with both.

This is NOT a cluster-validation check (that's Stage 1) -- it's a cheap
feasibility bridge before Stage 2's full classification pipeline: does
CGM share enough structure with the clinical variables to make predicting
cluster membership from CGM plausible at all? A strong top canonical
correlation says yes, and its loadings preview which CGM features are
likely to carry the signal Stage 2 will pick up on. A weak correlation
would be an early warning that Stage 2's AUROC may land near chance.

Outputs (in this directory):
  cca_canonical_correlations.csv
  cca_loadings.png -- top canonical variate pair scatter + loading bar charts
"""
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(OUT_DIR)
DATA_PATH = os.path.join(ROOT, "final_df.csv")
CGM_QC_MIN_PCT_ACTIVE = 70.0
N_COMPONENTS = 3

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
LOG_COLS = {CPEPTIDE_COL, INSULIN_COL}

df = pd.read_csv(DATA_PATH)
df.columns = [sanitize_column_name(c) for c in df.columns]
before = len(df)
df = df[df["qc_pct_active"] >= CGM_QC_MIN_PCT_ACTIVE]
df = df.dropna(subset=BASE_FEATURE_COLS + CGM_FEATURES_VIF_PRUNED).reset_index(drop=True)
print(f"n={len(df)} (dropped {before - len(df)} for QC/missingness)")

A = df[BASE_FEATURE_COLS].copy()
for c in LOG_COLS:
    A[c] = np.log1p(A[c])
A_scaled = StandardScaler().fit_transform(A)

B = df[CGM_FEATURES_VIF_PRUNED].copy()
B_scaled = StandardScaler().fit_transform(B)

n_components = min(N_COMPONENTS, len(BASE_FEATURE_COLS))
cca = CCA(n_components=n_components)
A_c, B_c = cca.fit_transform(A_scaled, B_scaled)

canonical_corrs = [np.corrcoef(A_c[:, i], B_c[:, i])[0, 1] for i in range(n_components)]
print("Canonical correlations:", [round(c, 3) for c in canonical_corrs])

pd.DataFrame({"component": range(1, n_components + 1), "canonical_correlation": canonical_corrs}).to_csv(
    os.path.join(OUT_DIR, "cca_canonical_correlations.csv"), index=False
)

# Loadings: correlation of each original variable with its own set's canonical variate
# (the standard way to interpret CCA components, analogous to PCA loadings).
a_loadings = pd.DataFrame(
    {f"CC{i+1}": [np.corrcoef(A_scaled[:, j], A_c[:, i])[0, 1] for j in range(A_scaled.shape[1])] for i in range(n_components)},
    index=[BASE_VAR_LABELS[c] for c in BASE_FEATURE_COLS],
)
b_loadings = pd.DataFrame(
    {f"CC{i+1}": [np.corrcoef(B_scaled[:, j], B_c[:, i])[0, 1] for j in range(B_scaled.shape[1])] for i in range(n_components)},
    index=CGM_FEATURES_VIF_PRUNED,
)
print("\nSet A (clinical) loadings on CC1:")
print(a_loadings["CC1"].sort_values(key=lambda s: s.abs(), ascending=False).round(3))
print("\nSet B (CGM) loadings on CC1 (top 8 by |loading|):")
print(b_loadings["CC1"].sort_values(key=lambda s: s.abs(), ascending=False).head(8).round(3))

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor="#fcfcfb")

ax = axes[0]
ax.set_facecolor("#fcfcfb")
ax.scatter(A_c[:, 0], B_c[:, 0], s=10, alpha=0.4, color="#2a78d6", edgecolor="none")
ax.set_xlabel("Clinical canonical variate 1 (Set A)", fontsize=10)
ax.set_ylabel("CGM canonical variate 1 (Set B)", fontsize=10)
ax.set_title(f"Top canonical variate pair\ncorrelation = {canonical_corrs[0]:.3f}", fontsize=12)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax = axes[1]
ax.set_facecolor("#fcfcfb")
vals = a_loadings["CC1"].sort_values()
colors = ["#d64550" if v < 0 else "#2a78d6" for v in vals]
ax.barh(vals.index, vals.values, color=colors)
ax.axvline(0, color="#999999", linewidth=1.0)
ax.set_title("Clinical variable loadings on CC1", fontsize=12)
ax.set_xlabel("Correlation with canonical variate")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax = axes[2]
ax.set_facecolor("#fcfcfb")
top_b = b_loadings["CC1"].sort_values(key=lambda s: s.abs(), ascending=False).head(10).sort_values()
colors = ["#d64550" if v < 0 else "#2a78d6" for v in top_b]
ax.barh(top_b.index, top_b.values, color=colors)
ax.axvline(0, color="#999999", linewidth=1.0)
ax.set_title("Top 10 CGM feature loadings on CC1", fontsize=12)
ax.set_xlabel("Correlation with canonical variate")
ax.tick_params(axis="y", labelsize=8)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.suptitle("CCA: clinical variables (Set A) vs curated CGM features (Set B)", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "cca_loadings.png"), dpi=200, facecolor=fig.get_facecolor())
plt.close(fig)
print("\nsaved cca_canonical_correlations.csv, cca_loadings.png")
