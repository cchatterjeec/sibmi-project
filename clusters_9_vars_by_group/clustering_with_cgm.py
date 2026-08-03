"""
Stage 0 (with CGM), stratified by study_group: 9-variable panel + 20 curated
CGM features (regression_no_correlated_features/select_cgm_features_vif.py),
refit within one study_group at a time. Requires qc_pct_active >= 70%.

Outputs (in clusters_9_vars_by_group/<group>/):
  clustering_with_cgm_assignments.csv
  clustering_with_cgm_pca_clusters_3d.png -- 2-panel: KMeans / Hierarchical
"""
import argparse
import os
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

sys.path.insert(0, os.path.join(C.ROOT, "regression_no_correlated_features"))
from select_cgm_features_vif import CGM_FEATURES_VIF_PRUNED

BASE_FEATURE_COLS = C.FEATURE_COLS
FEATURE_COLS = BASE_FEATURE_COLS + CGM_FEATURES_VIF_PRUNED


def main(slug):
    OUT_DIR = C.out_dir(slug)
    df = C.load_raw(require_cgm_qc=True)
    df = C.filter_group(df, slug)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    print(f"[{slug}] n={len(df)} ({len(BASE_FEATURE_COLS)} base + {len(CGM_FEATURES_VIF_PRUNED)} curated CGM features)")

    X = df[FEATURE_COLS].copy()
    for c in C.LOG_COLS:
        X[c] = np.log1p(X[c])
    X_scaled = StandardScaler().fit_transform(X)

    silhouettes = []
    for k in C.K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_scaled)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))
    best_k = list(C.K_RANGE)[int(np.argmax(silhouettes))]
    print(f"[{slug}] silhouette by k: {dict(zip(C.K_RANGE, [round(s, 3) for s in silhouettes]))}")
    print(f"[{slug}] chosen k = {best_k}")

    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit(X_scaled)
    df["kmeans_cluster"] = kmeans.labels_
    Z = linkage(X_scaled, method="ward")
    df["hier_cluster"] = fcluster(Z, t=best_k, criterion="maxclust") - 1
    ari = adjusted_rand_score(df["kmeans_cluster"], df["hier_cluster"])
    print(f"[{slug}] Adjusted Rand Index (KMeans vs Hierarchical) = {ari:.3f}")

    pca = PCA(n_components=3, random_state=0)
    X_pca = pca.fit_transform(X_scaled)
    df["pc1"], df["pc2"], df["pc3"] = X_pca[:, 0], X_pca[:, 1], X_pca[:, 2]
    var_explained = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(13, 6), facecolor="#fcfcfb")
    axes = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.set_xlabel(f"PC1 ({var_explained[0]:.0%})", fontsize=8.5)
        ax.set_ylabel(f"PC2 ({var_explained[1]:.0%})", fontsize=8.5)
        ax.set_zlabel(f"PC3 ({var_explained[2]:.0%})", fontsize=8.5)
        ax.view_init(elev=C.VIEW_ELEV, azim=C.VIEW_AZIM)

    for cl in range(best_k):
        sub = df[df["kmeans_cluster"] == cl]
        axes[0].scatter(sub["pc1"], sub["pc2"], sub["pc3"], s=8, alpha=0.5, color=C.CLUSTER_PALETTE[cl % len(C.CLUSTER_PALETTE)], label=f"Cluster {cl} (n={len(sub)})")
    axes[0].set_title(f"KMeans (k={best_k})", fontsize=11)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)

    for cl in range(best_k):
        sub = df[df["hier_cluster"] == cl]
        axes[1].scatter(sub["pc1"], sub["pc2"], sub["pc3"], s=8, alpha=0.5, color=C.CLUSTER_PALETTE[cl % len(C.CLUSTER_PALETTE)], label=f"Cluster {cl} (n={len(sub)})")
    axes[1].set_title(f"Hierarchical / Ward (k={best_k})", fontsize=11)
    axes[1].legend(loc="upper left", frameon=False, fontsize=8)

    fig.suptitle(f"Stage 0 (with CGM), study_group={C.GROUPS[slug]!r} (n={len(df)}): 9-var panel + {len(CGM_FEATURES_VIF_PRUNED)} CGM features", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "clustering_with_cgm_pca_clusters_3d.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)

    idc = C.id_col(df)
    df[[idc, "kmeans_cluster", "hier_cluster", "pc1", "pc2", "pc3"]].to_csv(
        os.path.join(OUT_DIR, "clustering_with_cgm_assignments.csv"), index=False
    )
    C.save_meta(slug, with_cgm_n=len(df), with_cgm_best_k=best_k)
    print(f"[{slug}] saved clustering_with_cgm_assignments.csv, clustering_with_cgm_pca_clusters_3d.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--group", required=True, choices=list(C.GROUPS))
    args = p.parse_args()
    main(args.group)
