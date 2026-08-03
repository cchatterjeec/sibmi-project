"""
3D version of clustering_by_study_group.py: same per-study_group clustering
(KMeans + Ward, run separately within each of Healthy/Pre-diabetes/Oral
medication/Insulin dependent, on age/HbA1c/BMI/C-peptide/insulin) -- but
PCA is fit with 3 components instead of 2, and each panel is a 3D scatter
(matplotlib mplot3d) instead of 2D.

Preprocessing and k selection (silhouette, swept k=2..6, per group) are
identical to clustering_by_study_group.py, so chosen k and cluster
assignments match it exactly -- only the PCA dimensionality/visualization
differ. This is a separate script -- it does not modify
clustering_by_study_group.py or its outputs.

Outputs (in this directory):
  study_group_pca_clusters_3d.png -- 2x4 grid of 3D scatter plots: KMeans
                                      (top row) / Hierarchical (bottom row),
                                      one column per study_group
  study_group_pca_loadings_3d.csv -- PC1/PC2/PC3 loadings per study_group
"""
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = "/n/groups/patel/chandrima/final_df.csv"
K_RANGE = range(2, 7)
VIEW_ELEV, VIEW_AZIM = 20, 45


def sanitize_column_name(col):
    return re.sub(r"[\[\]<]", "", col)


HBA1C_COL = sanitize_column_name("import_hba1c, Hemoglobin A1c/Hemoglobin.total in ")
CPEPTIDE_COL = sanitize_column_name("import_c_peptide, C peptide [Mass/volume] in Seru")
INSULIN_COL = sanitize_column_name("import_insulin, Insulin [Units/volume] in Serum o")
BMI_COL = "bmi_vsorres, BMI"

VAR_LABELS = {"age": "Age", HBA1C_COL: "HbA1c", BMI_COL: "BMI", CPEPTIDE_COL: "C-peptide", INSULIN_COL: "Insulin"}
FEATURE_COLS = ["age", HBA1C_COL, BMI_COL, CPEPTIDE_COL, INSULIN_COL]
LOG_COLS = {CPEPTIDE_COL, INSULIN_COL}

STUDY_GROUPS = [
    "healthy", "pre_diabetes_lifestyle_controlled",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled", "insulin_dependent",
]
GROUP_TITLES = {
    "healthy": "Healthy", "pre_diabetes_lifestyle_controlled": "Pre-diabetes",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled": "Oral medication",
    "insulin_dependent": "Insulin dependent",
}
CLUSTER_PALETTE = ["#2a78d6", "#eb6834", "#2ca858", "#a259c6", "#d6b02a", "#d64550"]

df = pd.read_csv(DATA_PATH)
df.columns = [sanitize_column_name(c) for c in df.columns]
df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
print(f"n={len(df)} (complete cases on {FEATURE_COLS})")

fig_pca = plt.figure(figsize=(22, 11), facecolor="#fcfcfb")
axes_pca = [[fig_pca.add_subplot(2, 4, row * 4 + col + 1, projection="3d") for col in range(4)] for row in range(2)]

all_loadings = []

for i, group in enumerate(STUDY_GROUPS):
    sub = df[df["study_group"] == group].copy()
    X = sub[FEATURE_COLS].copy()
    for c in LOG_COLS:
        X[c] = np.log1p(X[c])
    X_scaled = StandardScaler().fit_transform(X)

    silhouettes = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_scaled)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))
    best_k = list(K_RANGE)[int(np.argmax(silhouettes))]
    print(f"{GROUP_TITLES[group]} (n={len(sub)}): chosen k={best_k}")

    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit(X_scaled)
    sub["kmeans_cluster"] = kmeans.labels_
    Z = linkage(X_scaled, method="ward")
    sub["hier_cluster"] = fcluster(Z, t=best_k, criterion="maxclust") - 1

    pca = PCA(n_components=3, random_state=0)
    X_pca = pca.fit_transform(X_scaled)
    sub["pc1"], sub["pc2"], sub["pc3"] = X_pca[:, 0], X_pca[:, 1], X_pca[:, 2]
    var_explained = pca.explained_variance_ratio_
    print(f"  PCA variance explained: PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%}, "
          f"PC3={var_explained[2]:.1%} (cumulative {var_explained.sum():.1%})")

    loadings = pd.DataFrame(pca.components_.T, index=[VAR_LABELS[c] for c in FEATURE_COLS], columns=["PC1", "PC2", "PC3"])
    loadings.insert(0, "study_group", GROUP_TITLES[group])
    all_loadings.append(loadings.reset_index().rename(columns={"index": "variable"}))

    for row_idx, (method, col) in enumerate([("KMeans", "kmeans_cluster"), ("Hierarchical", "hier_cluster")]):
        ax = axes_pca[row_idx][i]
        ax.set_facecolor("#fcfcfb")
        for cl in sorted(sub[col].unique()):
            cl_sub = sub[sub[col] == cl]
            ax.scatter(cl_sub["pc1"], cl_sub["pc2"], cl_sub["pc3"], s=8, alpha=0.55,
                       color=CLUSTER_PALETTE[cl % len(CLUSTER_PALETTE)], label=f"Cluster {cl} (n={len(cl_sub)})")
        title_prefix = GROUP_TITLES[group] if row_idx == 0 else ""
        ax.set_title(f"{title_prefix}\n{method} (k={best_k})" if row_idx == 0 else f"{method} (k={best_k})", fontsize=10.5)
        ax.set_xlabel(f"PC1 ({var_explained[0]:.0%})", fontsize=7.5)
        ax.set_ylabel(f"PC2 ({var_explained[1]:.0%})", fontsize=7.5)
        ax.set_zlabel(f"PC3 ({var_explained[2]:.0%})", fontsize=7.5)
        ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
        ax.legend(loc="upper left", frameon=False, fontsize=6.5)

fig_pca.suptitle("Within-study_group clustering: 3D PCA projection fit separately per group", fontsize=14)
fig_pca.tight_layout()
fig_pca.savefig(os.path.join(OUT_DIR, "study_group_pca_clusters_3d.png"), dpi=200, facecolor=fig_pca.get_facecolor())
plt.close(fig_pca)

loadings_df = pd.concat(all_loadings, ignore_index=True)
loadings_df.to_csv(os.path.join(OUT_DIR, "study_group_pca_loadings_3d.csv"), index=False)
print("\nsaved study_group_pca_clusters_3d.png, study_group_pca_loadings_3d.csv")
