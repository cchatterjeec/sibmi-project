"""
KMeans and hierarchical (Ward) clustering on the same five variables as
clustering_analysis.py (age, HbA1c, BMI, C-peptide, insulin), using the
full cohort (n=2217, no CGM QC gate -- none of these variables are
CGM-derived) -- but this time PCA is fit with 3 components instead of 2,
and visualized as a 3D scatter (matplotlib mplot3d) instead of a 2D one.

This is a separate script from clustering_analysis.py -- it does not
overwrite or replace it or its outputs. Preprocessing (log1p on insulin/
C-peptide, then z-score all 5 features) and k selection (silhouette,
swept k=2..8) are identical to clustering_analysis.py, so the chosen k
and cluster assignments should match it exactly -- only the PCA
dimensionality and visualization differ.

A 3D static plot only shows one fixed viewing angle (elev=20, azim=45
here) -- some cluster separation visible from other angles won't be
visible in a static PNG.

Outputs (in this directory):
  pca_3d_loadings.csv       -- PC1/PC2/PC3 loadings for each of the 5 variables
  pca_3d_clusters.png       -- 2x2 grid of 3D scatter plots: KMeans / Hierarchical /
                               study_group / HbA1c group
  pca_3d_cluster_assignments.csv -- participant_id, KMeans cluster, Hierarchical
                               cluster, study_group, hba1c_group, pc1/pc2/pc3
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
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = "/n/groups/patel/chandrima/final_df.csv"
K_RANGE = range(2, 9)
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

CLUSTER_PALETTE = ["#2a78d6", "#eb6834", "#2ca858", "#a259c6", "#d6b02a", "#d64550"]
HBA1C_GROUP_COLORS = {"Normal (<5.7%)": "#2a78d6", "Prediabetes (5.7-6.4%)": "#d6b02a", "Diabetes (>=6.5%)": "#d64550"}


def main():
    df = pd.read_csv(DATA_PATH)
    df.columns = [sanitize_column_name(c) for c in df.columns]
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    print(f"n={len(df)} (complete cases on {FEATURE_COLS})")

    X = df[FEATURE_COLS].copy()
    for c in LOG_COLS:
        X[c] = np.log1p(X[c])
    X_scaled = StandardScaler().fit_transform(X)

    silhouettes = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_scaled)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))
    best_k = list(K_RANGE)[int(np.argmax(silhouettes))]
    print(f"silhouette by k: {dict(zip(K_RANGE, [round(s, 3) for s in silhouettes]))}")
    print(f"chosen k (max silhouette) = {best_k}")

    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit(X_scaled)
    df["kmeans_cluster"] = kmeans.labels_
    Z = linkage(X_scaled, method="ward")
    df["hier_cluster"] = fcluster(Z, t=best_k, criterion="maxclust") - 1
    ari = adjusted_rand_score(df["kmeans_cluster"], df["hier_cluster"])
    print(f"Adjusted Rand Index (KMeans vs Hierarchical agreement) = {ari:.3f}")

    # --- PCA with 3 components ---
    pca = PCA(n_components=3, random_state=0)
    X_pca = pca.fit_transform(X_scaled)
    df["pc1"], df["pc2"], df["pc3"] = X_pca[:, 0], X_pca[:, 1], X_pca[:, 2]
    var_explained = pca.explained_variance_ratio_
    print(f"PCA variance explained: PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%}, "
          f"PC3={var_explained[2]:.1%} (cumulative {var_explained.sum():.1%})")

    loadings = pd.DataFrame(pca.components_.T, index=[VAR_LABELS[c] for c in FEATURE_COLS], columns=["PC1", "PC2", "PC3"])
    loadings.to_csv(os.path.join(OUT_DIR, "pca_3d_loadings.csv"))
    print("\nPCA loadings (weight of each variable on PC1/PC2/PC3):")
    print(loadings.round(3).to_string())

    hba1c_bins = [-np.inf, 5.7, 6.5, np.inf]
    hba1c_group_labels = ["Normal (<5.7%)", "Prediabetes (5.7-6.4%)", "Diabetes (>=6.5%)"]
    df["hba1c_group"] = pd.cut(df[HBA1C_COL], bins=hba1c_bins, labels=hba1c_group_labels, right=False)
    study_groups = sorted(df["study_group"].dropna().unique().tolist())

    fig = plt.figure(figsize=(16, 15), facecolor="#fcfcfb")
    axes = [fig.add_subplot(2, 2, i + 1, projection="3d") for i in range(4)]
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.set_xlabel(f"PC1 ({var_explained[0]:.0%})", fontsize=9)
        ax.set_ylabel(f"PC2 ({var_explained[1]:.0%})", fontsize=9)
        ax.set_zlabel(f"PC3 ({var_explained[2]:.0%})", fontsize=9)
        ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    for cl in range(best_k):
        sub = df[df["kmeans_cluster"] == cl]
        axes[0].scatter(sub["pc1"], sub["pc2"], sub["pc3"], s=8, alpha=0.5,
                         color=CLUSTER_PALETTE[cl % len(CLUSTER_PALETTE)], label=f"Cluster {cl} (n={len(sub)})")
    axes[0].set_title(f"KMeans (k={best_k})", fontsize=12)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)

    for cl in range(best_k):
        sub = df[df["hier_cluster"] == cl]
        axes[1].scatter(sub["pc1"], sub["pc2"], sub["pc3"], s=8, alpha=0.5,
                         color=CLUSTER_PALETTE[cl % len(CLUSTER_PALETTE)], label=f"Cluster {cl} (n={len(sub)})")
    axes[1].set_title(f"Hierarchical / Ward (k={best_k})", fontsize=12)
    axes[1].legend(loc="upper left", frameon=False, fontsize=8)

    for i, sg in enumerate(study_groups):
        sub = df[df["study_group"] == sg]
        axes[2].scatter(sub["pc1"], sub["pc2"], sub["pc3"], s=8, alpha=0.5,
                         color=CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)], label=f"{sg} (n={len(sub)})")
    axes[2].set_title("Actual study_group", fontsize=12)
    axes[2].legend(loc="upper left", frameon=False, fontsize=7)

    for lv in hba1c_group_labels:
        sub = df[df["hba1c_group"] == lv]
        axes[3].scatter(sub["pc1"], sub["pc2"], sub["pc3"], s=8, alpha=0.5,
                         color=HBA1C_GROUP_COLORS[lv], label=f"{lv} (n={len(sub)})")
    axes[3].set_title("HbA1c group (ADA thresholds)", fontsize=12)
    axes[3].legend(loc="upper left", frameon=False, fontsize=8)

    fig.suptitle("3D PCA projection (age, HbA1c, BMI, C-peptide, insulin)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pca_3d_clusters.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)

    id_col = "participant_id" if "participant_id" in df.columns else df.columns[0]
    df[[id_col, "kmeans_cluster", "hier_cluster", "study_group", "hba1c_group", "pc1", "pc2", "pc3"]].to_csv(
        os.path.join(OUT_DIR, "pca_3d_cluster_assignments.csv"), index=False
    )
    print("\nsaved pca_3d_loadings.csv, pca_3d_clusters.png, pca_3d_cluster_assignments.csv")


if __name__ == "__main__":
    main()
