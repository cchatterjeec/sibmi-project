"""
Stage 0 (with CGM): KMeans + hierarchical (Ward) clustering on the 5 base
variables + 20 curated CGM features (regression_no_correlated_features/
select_cgm_features_vif.py) -- same methodology as
cluster/clustering_analysis_with_cgm.py, refit fresh here so this
directory is self-contained. PCA is fit with 3 components (not 2).

Requires the standard qc_pct_active >= 70% CGM quality filter, since CGM
features are part of the feature set here (unlike clustering_base.py).

Outputs (in this directory):
  clustering_with_cgm_assignments.csv -- participant_id, kmeans_cluster,
                                          hier_cluster, study_group, pc1, pc2, pc3
  clustering_with_cgm_pca_clusters_3d.png -- 3D scatter: KMeans / Hierarchical / study_group
"""
import os
import re
import sys

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
ROOT = os.path.dirname(OUT_DIR)
DATA_PATH = os.path.join(ROOT, "final_df.csv")
K_RANGE = range(2, 9)
CGM_QC_MIN_PCT_ACTIVE = 70.0
VIEW_ELEV, VIEW_AZIM = 20, 45

sys.path.insert(0, os.path.join(ROOT, "regression_no_correlated_features"))
from select_cgm_features_vif import CGM_FEATURES_VIF_PRUNED


def sanitize_column_name(col):
    return re.sub(r"[\[\]<]", "", col)


HBA1C_COL = sanitize_column_name("import_hba1c, Hemoglobin A1c/Hemoglobin.total in ")
CPEPTIDE_COL = sanitize_column_name("import_c_peptide, C peptide [Mass/volume] in Seru")
INSULIN_COL = sanitize_column_name("import_insulin, Insulin [Units/volume] in Serum o")
BMI_COL = "bmi_vsorres, BMI"

BASE_FEATURE_COLS = ["age", HBA1C_COL, BMI_COL, CPEPTIDE_COL, INSULIN_COL]
FEATURE_COLS = BASE_FEATURE_COLS + CGM_FEATURES_VIF_PRUNED
LOG_COLS = {CPEPTIDE_COL, INSULIN_COL}
CLUSTER_PALETTE = ["#2a78d6", "#eb6834", "#2ca858", "#a259c6", "#d6b02a", "#d64550"]


def main():
    df = pd.read_csv(DATA_PATH)
    df.columns = [sanitize_column_name(c) for c in df.columns]
    before = len(df)
    df = df[df["qc_pct_active"] >= CGM_QC_MIN_PCT_ACTIVE]
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    print(f"n={len(df)} (dropped {before - len(df)} for QC/missingness; "
          f"{len(BASE_FEATURE_COLS)} base + {len(CGM_FEATURES_VIF_PRUNED)} curated CGM features)")

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

    fig.suptitle(f"Stage 0 (with CGM): 3D PCA projection (5 base vars + {len(CGM_FEATURES_VIF_PRUNED)} curated CGM features)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "clustering_with_cgm_pca_clusters_3d.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)

    id_col = "participant_id" if "participant_id" in df.columns else df.columns[0]
    df[[id_col, "kmeans_cluster", "hier_cluster", "study_group", "pc1", "pc2", "pc3"]].to_csv(
        os.path.join(OUT_DIR, "clustering_with_cgm_assignments.csv"), index=False
    )
    print("saved clustering_with_cgm_assignments.csv, clustering_with_cgm_pca_clusters_3d.png")


if __name__ == "__main__":
    main()
