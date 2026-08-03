"""
Stage 1c: Gaussian Mixture Model (soft clustering) vs KMeans, same
standardized 9-variable space. Compares hard labels (ARI + crosstab) and
flags people with max posterior probability <0.6 as ambiguous.

Outputs (in this directory):
  gmm_vs_kmeans_comparison.png
  gmm_ambiguous_participants.csv
"""
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = "/n/groups/patel/chandrima/final_df.csv"
BEST_K = 3
AMBIGUOUS_THRESHOLD = 0.6
VIEW_ELEV, VIEW_AZIM = 20, 45
CLUSTER_PALETTE = ["#2a78d6", "#eb6834", "#2ca858"]


def sanitize_column_name(col):
    return re.sub(r"[\[\]<]", "", col)


CPEPTIDE_COL = sanitize_column_name("import_c_peptide, C peptide [Mass/volume] in Seru")
INSULIN_COL = sanitize_column_name("import_insulin, Insulin [Units/volume] in Serum o")
BMI_COL = "bmi_vsorres, BMI"
CREATININE_COL = sanitize_column_name("import_creatinine, Creatinine [Mass/volume] in Se")
URINE_ALBUMIN_COL = sanitize_column_name("import_urine_albumin, Albumin [Mass/volume] in Ur")
URINE_CREATININE_COL = sanitize_column_name("import_urine_creatinine, Creatinine [Mass/volume]")
TRIGLYCERIDES_COL = sanitize_column_name("import_triglycerides, Triglyceride [Mass/volume] ")
HDL_COL = sanitize_column_name("import_hdl_cholesterol, Cholesterol in HDL [Mass/")
CRP_COL = sanitize_column_name("import_crp_hs, C reactive protein [Mass/volume] i")
FEATURE_COLS = ["age", BMI_COL, INSULIN_COL, CPEPTIDE_COL, CREATININE_COL, "uacr", TRIGLYCERIDES_COL, HDL_COL, CRP_COL]
LOG_COLS = {INSULIN_COL, CPEPTIDE_COL, CREATININE_COL, "uacr", TRIGLYCERIDES_COL, CRP_COL}

df = pd.read_csv(DATA_PATH)
df.columns = [sanitize_column_name(c) for c in df.columns]
df["uacr"] = df[URINE_ALBUMIN_COL] / df[URINE_CREATININE_COL].replace(0, np.nan)
df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

assignments = pd.read_csv(os.path.join(OUT_DIR, "clustering_base_assignments.csv"))
id_col = "participant_id" if "participant_id" in df.columns else df.columns[0]
df = df.merge(assignments[[id_col, "kmeans_cluster", "pc1", "pc2", "pc3"]], on=id_col)
print(f"n={len(df)}")

X = df[FEATURE_COLS].copy()
for c in LOG_COLS:
    X[c] = np.log1p(X[c])
X_scaled = StandardScaler().fit_transform(X)

gmm = GaussianMixture(n_components=BEST_K, random_state=0, n_init=10).fit(X_scaled)
gmm_labels = gmm.predict(X_scaled)
gmm_proba = gmm.predict_proba(X_scaled)
df["gmm_cluster"] = gmm_labels
df["gmm_max_proba"] = gmm_proba.max(axis=1)

ari = adjusted_rand_score(df["kmeans_cluster"], df["gmm_cluster"])
print(f"Adjusted Rand Index (KMeans vs GMM) = {ari:.3f}")

crosstab = pd.crosstab(df["kmeans_cluster"], df["gmm_cluster"])
print("\nKMeans (rows) vs GMM (cols) crosstab:")
print(crosstab)

ambiguous = df[df["gmm_max_proba"] < AMBIGUOUS_THRESHOLD]
print(f"\n{len(ambiguous)} of {len(df)} participants have max GMM posterior < {AMBIGUOUS_THRESHOLD} (ambiguous membership)")
ambiguous[[id_col, "kmeans_cluster", "gmm_cluster", "gmm_max_proba"]].to_csv(
    os.path.join(OUT_DIR, "gmm_ambiguous_participants.csv"), index=False
)

fig = plt.figure(figsize=(16, 6), facecolor="#fcfcfb")
ax0 = fig.add_subplot(1, 3, 1)
ax0.set_facecolor("#fcfcfb")
im = ax0.imshow(crosstab.values, cmap="Blues")
for i in range(crosstab.shape[0]):
    for j in range(crosstab.shape[1]):
        ax0.text(j, i, str(crosstab.values[i, j]), ha="center", va="center",
                  color="white" if crosstab.values[i, j] > crosstab.values.max() / 2 else "black")
ax0.set_xticks(range(crosstab.shape[1])); ax0.set_xticklabels([f"GMM {c}" for c in crosstab.columns])
ax0.set_yticks(range(crosstab.shape[0])); ax0.set_yticklabels([f"KMeans {c}" for c in crosstab.index])
ax0.set_title(f"Crosstab (ARI={ari:.3f})", fontsize=11)

ax1 = fig.add_subplot(1, 3, 2, projection="3d")
ax2 = fig.add_subplot(1, 3, 3, projection="3d")
for ax, col, title in [(ax1, "kmeans_cluster", "KMeans"), (ax2, "gmm_cluster", "GMM")]:
    ax.set_facecolor("#fcfcfb")
    for cl in sorted(df[col].unique()):
        sub = df[df[col] == cl]
        ax.scatter(sub["pc1"], sub["pc2"], sub["pc3"], s=8, alpha=0.5, color=CLUSTER_PALETTE[cl % len(CLUSTER_PALETTE)], label=f"Cluster {cl} (n={len(sub)})")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("PC1", fontsize=8); ax.set_ylabel("PC2", fontsize=8); ax.set_zlabel("PC3", fontsize=8)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.legend(loc="upper left", frameon=False, fontsize=7)

fig.suptitle("KMeans vs Gaussian Mixture Model, same standardized 9-var space", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "gmm_vs_kmeans_comparison.png"), dpi=200, facecolor=fig.get_facecolor())
plt.close(fig)
print("saved gmm_vs_kmeans_comparison.png, gmm_ambiguous_participants.csv")
