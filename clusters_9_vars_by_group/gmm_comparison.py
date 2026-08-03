"""
Stage 1c, stratified: Gaussian Mixture Model (soft clustering) vs KMeans,
within-group 9-variable space, k = meta.json's best_k. Flags people with max
posterior probability < 0.6 as ambiguous.

Outputs: gmm_vs_kmeans_comparison.png, gmm_ambiguous_participants.csv
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

AMBIGUOUS_THRESHOLD = 0.6


def main(slug):
    OUT_DIR = C.out_dir(slug)
    meta = C.load_meta(slug)
    best_k = meta["best_k"]
    target = meta["target_cluster_kmeans"]

    df = C.load_raw()
    df = C.filter_group(df, slug)
    df = df.dropna(subset=C.FEATURE_COLS).reset_index(drop=True)

    assignments = pd.read_csv(os.path.join(OUT_DIR, "clustering_base_assignments.csv"))
    idc = C.id_col(df)
    df = df.merge(assignments[[idc, "kmeans_cluster", "pc1", "pc2", "pc3"]], on=idc)
    print(f"[{slug}] n={len(df)}, k={best_k}")

    X = df[C.FEATURE_COLS].copy()
    for c in C.LOG_COLS:
        X[c] = np.log1p(X[c])
    X_scaled = StandardScaler().fit_transform(X)

    gmm = GaussianMixture(n_components=best_k, random_state=0, n_init=10).fit(X_scaled)
    gmm_labels = gmm.predict(X_scaled)
    gmm_proba = gmm.predict_proba(X_scaled)
    df["gmm_cluster"] = gmm_labels
    df["gmm_max_proba"] = gmm_proba.max(axis=1)

    ari = adjusted_rand_score(df["kmeans_cluster"], df["gmm_cluster"])
    print(f"[{slug}] Adjusted Rand Index (KMeans vs GMM) = {ari:.3f}")

    crosstab = pd.crosstab(df["kmeans_cluster"], df["gmm_cluster"])
    print(f"[{slug}] KMeans (rows) vs GMM (cols) crosstab:")
    print(crosstab)

    ambiguous = df[df["gmm_max_proba"] < AMBIGUOUS_THRESHOLD]
    print(f"[{slug}] {len(ambiguous)} of {len(df)} participants have max GMM posterior < {AMBIGUOUS_THRESHOLD}")
    ambiguous[[idc, "kmeans_cluster", "gmm_cluster", "gmm_max_proba"]].to_csv(
        os.path.join(OUT_DIR, "gmm_ambiguous_participants.csv"), index=False
    )

    fig = plt.figure(figsize=(16, 6), facecolor="#fcfcfb")
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.set_facecolor("#fcfcfb")
    ax0.imshow(crosstab.values, cmap="Blues")
    for i in range(crosstab.shape[0]):
        for j in range(crosstab.shape[1]):
            ax0.text(j, i, str(crosstab.values[i, j]), ha="center", va="center",
                      color="white" if crosstab.values[i, j] > crosstab.values.max() / 2 else "black")
    ax0.set_xticks(range(crosstab.shape[1])); ax0.set_xticklabels([f"GMM {c}" for c in crosstab.columns])
    row_labels = [f"KMeans {c}" + (" (kidney)" if c == target else "") for c in crosstab.index]
    ax0.set_yticks(range(crosstab.shape[0])); ax0.set_yticklabels(row_labels)
    ax0.set_title(f"Crosstab (ARI={ari:.3f})", fontsize=11)

    ax1 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    for ax, col, title in [(ax1, "kmeans_cluster", "KMeans"), (ax2, "gmm_cluster", "GMM")]:
        ax.set_facecolor("#fcfcfb")
        for cl in sorted(df[col].unique()):
            sub = df[df[col] == cl]
            ax.scatter(sub["pc1"], sub["pc2"], sub["pc3"], s=8, alpha=0.5, color=C.CLUSTER_PALETTE[cl % len(C.CLUSTER_PALETTE)], label=f"Cluster {cl} (n={len(sub)})")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("PC1", fontsize=8); ax.set_ylabel("PC2", fontsize=8); ax.set_zlabel("PC3", fontsize=8)
        ax.view_init(elev=C.VIEW_ELEV, azim=C.VIEW_AZIM)
        ax.legend(loc="upper left", frameon=False, fontsize=7)

    fig.suptitle(f"{slug}: KMeans vs Gaussian Mixture Model, same standardized 9-var space", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "gmm_vs_kmeans_comparison.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[{slug}] saved gmm_vs_kmeans_comparison.png, gmm_ambiguous_participants.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--group", required=True, choices=list(C.GROUPS))
    args = p.parse_args()
    main(args.group)
