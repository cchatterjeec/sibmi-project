"""
Stage 0 (base), stratified by study_group: same 9-variable diabetes-relevant
panel and preprocessing as clusters_9_vars/clustering_base.py, refit
separately WITHIN one study_group at a time (--group healthy|pre_diabetes|
oral_medication|insulin_dependent) instead of pooling the whole cohort.

Since k and cluster order aren't guaranteed to reproduce per stratum, the
"kidney-complication" target cluster is auto-identified per group as whichever
resulting KMeans cluster has the highest combined z-score of mean Creatinine +
mean UACR (see common.pick_kidney_cluster) -- not assumed to be index 2.

Outputs (in clusters_9_vars_by_group/<group>/):
  clustering_base_assignments.csv
  clustering_base_pca_clusters_3d.png -- 2-panel: KMeans / Hierarchical
  clustering_base_loadings.csv
  clustering_base_cluster_profiles.csv
  meta.json -- n, best_k, target_cluster_kmeans, target_cluster_hier
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


def main(slug):
    OUT_DIR = C.out_dir(slug)
    df = C.load_raw()
    df = C.filter_group(df, slug)
    df = df.dropna(subset=C.FEATURE_COLS).reset_index(drop=True)
    print(f"[{slug}] n={len(df)} (complete cases on {C.FEATURE_COLS})")

    X = df[C.FEATURE_COLS].copy()
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
    print(f"[{slug}] PCA variance explained: PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%}, "
          f"PC3={var_explained[2]:.1%} (cumulative {var_explained.sum():.1%})")

    loadings = pd.DataFrame(pca.components_.T, index=[C.VAR_LABELS[c] for c in C.FEATURE_COLS], columns=["PC1", "PC2", "PC3"])
    loadings.to_csv(os.path.join(OUT_DIR, "clustering_base_loadings.csv"))

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

    fig.suptitle(f"Stage 0 (base), study_group={C.GROUPS[slug]!r} (n={len(df)}): 3D PCA, 9-var panel, no HbA1c/glucose", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "clustering_base_pca_clusters_3d.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)

    profiles = []
    for method, col in [("KMeans", "kmeans_cluster"), ("Hierarchical", "hier_cluster")]:
        for cl in sorted(df[col].unique()):
            sub = df[df[col] == cl]
            row = {"method": method, "cluster": cl, "n": len(sub)}
            for c in C.FEATURE_COLS:
                row[C.VAR_LABELS[c]] = sub[c].mean()
            profiles.append(row)
    profiles_df = pd.DataFrame(profiles)
    profiles_df.to_csv(os.path.join(OUT_DIR, "clustering_base_cluster_profiles.csv"), index=False)
    print(f"[{slug}] cluster profiles:")
    print(profiles_df.to_string(index=False))

    target_kmeans, ranked_kmeans = C.pick_kidney_cluster(profiles_df, "KMeans")
    target_hier, ranked_hier = C.pick_kidney_cluster(profiles_df, "Hierarchical")
    print(f"[{slug}] auto-identified kidney-complication cluster: KMeans={target_kmeans}, Hierarchical={target_hier}")
    print(ranked_kmeans.to_string(index=False))

    idc = C.id_col(df)
    df[[idc, "kmeans_cluster", "hier_cluster", "pc1", "pc2", "pc3"]].to_csv(
        os.path.join(OUT_DIR, "clustering_base_assignments.csv"), index=False
    )
    C.save_meta(
        slug,
        study_group=C.GROUPS[slug],
        n=len(df),
        best_k=best_k,
        ari_kmeans_vs_hier=round(float(ari), 4),
        target_cluster_kmeans=target_kmeans,
        target_cluster_hier=target_hier,
        target_cluster_kmeans_n=int(ranked_kmeans.iloc[0]["n"]),
    )
    print(f"[{slug}] saved clustering_base_assignments.csv, clustering_base_pca_clusters_3d.png, "
          "clustering_base_loadings.csv, clustering_base_cluster_profiles.csv, meta.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--group", required=True, choices=list(C.GROUPS))
    args = p.parse_args()
    main(args.group)
