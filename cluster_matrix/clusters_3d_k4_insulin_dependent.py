"""
3D PCA scatter of the insulin_dependent group's forced-k=4 clustering (same
fit as cgm_predicts_cluster_k4_insulin_dependent.py's fit_k4_clusters --
spectral affinity + KMeans, k forced to 4 instead of the silhouette-chosen
k=2 in cluster_assignments.csv). That script only ever saved participant_id
and cluster; this refits the identical, deterministic (random_state=0) k=4
fit and adds pc1-3 in the same standardized 5-feature PCA space that
clustering_matrix.py uses for its k=2 plot, so the k=4 clusters can be
viewed in 3D the same way.

Outputs (in cluster_matrix/insulin_dependent/):
  cluster_assignments_k4.csv -- overwritten with pc1-3 columns added
  clusters_3d_k4.png
"""
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

CM_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("cm", os.path.join(CM_DIR, "clustering_matrix.py"))
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)
C = cm.C

SLUG = "insulin_dependent"
K_FORCED = 4


def main():
    df = C.load_raw()
    df = C.filter_group(df, SLUG)
    df = df.dropna(subset=C.FEATURE_COLS).reset_index(drop=True)

    X = df[C.FEATURE_COLS].copy()
    for c in C.LOG_COLS:
        X[c] = np.log1p(X[c])

    idc = C.id_col(df)
    df, X, outliers_df = cm.flag_and_remove_outliers(df, X, idc)
    print(f"[{SLUG}] n={len(df)} after removing {len(outliers_df)} outlier(s)")

    X_scaled = StandardScaler().fit_transform(X)
    affinity, sigma, n_neighbors, n_components = cm.connected_neighbor_graph(X_scaled)
    emb = cm.spectral_embedding(affinity, K_FORCED)
    raw_labels = KMeans(n_clusters=K_FORCED, n_init=10, random_state=0).fit_predict(emb)

    composite_idx = [C.FEATURE_COLS.index(c) for c in (C.BMI_COL, C.INSULIN_COL, C.CPEPTIDE_COL)]
    composite_score = X_scaled[:, composite_idx].mean(axis=1)
    cluster_order = pd.Series(composite_score).groupby(raw_labels).mean().sort_values().index.tolist()
    relabel_map = {old: new for new, old in enumerate(cluster_order)}
    df["cluster"] = pd.Series(raw_labels).map(relabel_map).values

    pca = PCA(n_components=3, random_state=0)
    X_pca = pca.fit_transform(X_scaled)
    df["pc1"], df["pc2"], df["pc3"] = X_pca[:, 0], X_pca[:, 1], X_pca[:, 2]
    var_explained = pca.explained_variance_ratio_
    print(f"[{SLUG}] PCA variance explained: PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%}, "
          f"PC3={var_explained[2]:.1%} (cumulative {var_explained.sum():.1%})")

    out_dir = C.out_dir(SLUG)
    assign_path = os.path.join(out_dir, "cluster_assignments_k4.csv")
    df[[idc, "cluster", "pc1", "pc2", "pc3"]].to_csv(assign_path, index=False)
    print(f"saved {assign_path}")

    fig = plt.figure(figsize=(7.5, 6.5), facecolor="#fcfcfb")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#fcfcfb")
    ax.set_xlabel(f"PC1 ({var_explained[0]:.0%})", fontsize=9)
    ax.set_ylabel(f"PC2 ({var_explained[1]:.0%})", fontsize=9)
    ax.set_zlabel(f"PC3 ({var_explained[2]:.0%})", fontsize=9)
    ax.view_init(elev=C.VIEW_ELEV, azim=C.VIEW_AZIM)
    for cl in range(K_FORCED):
        sub = df[df["cluster"] == cl]
        ax.scatter(
            sub["pc1"], sub["pc2"], sub["pc3"], s=10, alpha=0.55,
            color=C.CLUSTER_PALETTE[cl % len(C.CLUSTER_PALETTE)],
            label=f"Cluster {cl} (n={len(sub)})",
        )
    ax.set_title(
        f"{SLUG}: spectral affinity + KMeans (k={K_FORCED} forced) on\n"
        f"age/BMI/insulin/C-peptide/HbA1c, n={len(df)}", fontsize=10,
    )
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "clusters_3d_k4.png")
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
