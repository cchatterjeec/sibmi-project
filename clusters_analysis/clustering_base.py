"""
Stage 0 (base): KMeans + hierarchical (Ward) clustering on the 5 base
variables (age, HbA1c, BMI, C-peptide, insulin), full cohort, no CGM
features -- same methodology as cluster/clustering_analysis.py, refit
fresh here so this new directory is self-contained. PCA is fit with 3
components (not 2), and the saved embedding (pc1/pc2/pc3) is what every
other script in this directory reads from, so everything downstream shares
the same coordinates.

Preprocessing: insulin/C-peptide log1p-transformed (right-skewed lab
concentrations), then all 5 features z-scored. k chosen by silhouette,
swept k=2..8.

Outputs (in this directory):
  clustering_base_assignments.csv -- participant_id, kmeans_cluster,
                                      hier_cluster, study_group, pc1, pc2, pc3
  clustering_base_pca_clusters_3d.png -- 3D scatter: KMeans / Hierarchical / study_group
  clustering_base_loadings.csv    -- PC1/PC2/PC3 loadings
"""
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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
    print(f"chosen k = {best_k}")

    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit(X_scaled)
    df["kmeans_cluster"] = kmeans.labels_
    Z = linkage(X_scaled, method="ward")
    df["hier_cluster"] = fcluster(Z, t=best_k, criterion="maxclust") - 1
    ari = adjusted_rand_score(df["kmeans_cluster"], df["hier_cluster"])
    print(f"Adjusted Rand Index (KMeans vs Hierarchical) = {ari:.3f}")

    pca = PCA(n_components=3, random_state=0)
    X_pca = pca.fit_transform(X_scaled)
    df["pc1"], df["pc2"], df["pc3"] = X_pca[:, 0], X_pca[:, 1], X_pca[:, 2]
    var_explained = pca.explained_variance_ratio_
    print(f"PCA variance explained: PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%}, "
          f"PC3={var_explained[2]:.1%} (cumulative {var_explained.sum():.1%})")

    loadings = pd.DataFrame(pca.components_.T, index=[VAR_LABELS[c] for c in FEATURE_COLS], columns=["PC1", "PC2", "PC3"])
    loadings.to_csv(os.path.join(OUT_DIR, "clustering_base_loadings.csv"))
    print(loadings.round(3).to_string())

    study_groups = sorted(df["study_group"].dropna().unique().tolist())
    fig = plt.figure(figsize=(19, 6), facecolor="#fcfcfb")
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.set_xlabel(f"PC1 ({var_explained[0]:.0%})", fontsize=8.5)
        ax.set_ylabel(f"PC2 ({var_explained[1]:.0%})", fontsize=8.5)
        ax.set_zlabel(f"PC3 ({var_explained[2]:.0%})", fontsize=8.5)
        ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    for cl in range(best_k):
        sub = df[df["kmeans_cluster"] == cl]
        axes[0].scatter(sub["pc1"], sub["pc2"], sub["pc3"], s=8, alpha=0.5, color=CLUSTER_PALETTE[cl % len(CLUSTER_PALETTE)], label=f"Cluster {cl} (n={len(sub)})")
    axes[0].set_title(f"KMeans (k={best_k})", fontsize=11)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)

    for cl in range(best_k):
        sub = df[df["hier_cluster"] == cl]
        axes[1].scatter(sub["pc1"], sub["pc2"], sub["pc3"], s=8, alpha=0.5, color=CLUSTER_PALETTE[cl % len(CLUSTER_PALETTE)], label=f"Cluster {cl} (n={len(sub)})")
    axes[1].set_title(f"Hierarchical / Ward (k={best_k})", fontsize=11)
    axes[1].legend(loc="upper left", frameon=False, fontsize=8)

    for i, sg in enumerate(study_groups):
        sub = df[df["study_group"] == sg]
        axes[2].scatter(sub["pc1"], sub["pc2"], sub["pc3"], s=8, alpha=0.5, color=CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)], label=f"{sg} (n={len(sub)})")
    axes[2].set_title("Actual study_group", fontsize=11)
    axes[2].legend(loc="upper left", frameon=False, fontsize=7)

    fig.suptitle("Stage 0 (base): 3D PCA projection (age, HbA1c, BMI, C-peptide, insulin)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "clustering_base_pca_clusters_3d.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)

    id_col = "participant_id" if "participant_id" in df.columns else df.columns[0]
    df[[id_col, "kmeans_cluster", "hier_cluster", "study_group", "pc1", "pc2", "pc3"]].to_csv(
        os.path.join(OUT_DIR, "clustering_base_assignments.csv"), index=False
    )
    print("saved clustering_base_assignments.csv, clustering_base_pca_clusters_3d.png, clustering_base_loadings.csv")


if __name__ == "__main__":
    main()
