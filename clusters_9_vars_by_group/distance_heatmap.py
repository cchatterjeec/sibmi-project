"""
Stage 1b, stratified: pairwise Euclidean distance matrix on the standardized
within-group 9-variable space, rows/columns sorted by KMeans cluster label.

Output: distance_heatmap.png
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main(slug):
    OUT_DIR = C.out_dir(slug)
    meta = C.load_meta(slug)
    target = meta["target_cluster_kmeans"]

    df = C.load_raw()
    df = C.filter_group(df, slug)
    df = df.dropna(subset=C.FEATURE_COLS).reset_index(drop=True)

    assignments = pd.read_csv(os.path.join(OUT_DIR, "clustering_base_assignments.csv"))
    idc = C.id_col(df)
    df = df.merge(assignments[[idc, "kmeans_cluster"]], on=idc)
    print(f"[{slug}] n={len(df)}")

    X = df[C.FEATURE_COLS].copy()
    for c in C.LOG_COLS:
        X[c] = np.log1p(X[c])
    X_scaled = StandardScaler().fit_transform(X)

    order = df.sort_values("kmeans_cluster").index.to_numpy()
    dist_matrix = squareform(pdist(X_scaled, metric="euclidean"))
    dist_sorted = dist_matrix[np.ix_(order, order)]
    cluster_sorted = df.loc[order, "kmeans_cluster"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.5, 7.5), facecolor="#fcfcfb")
    im = ax.imshow(dist_sorted, cmap="viridis", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Euclidean distance (standardized 9-var space)", fontsize=10)

    boundaries = np.where(np.diff(cluster_sorted) != 0)[0] + 1
    for b in boundaries:
        ax.axhline(b, color="white", linewidth=1.2)
        ax.axvline(b, color="white", linewidth=1.2)

    tick_positions, tick_labels = [], []
    start = 0
    for cl in sorted(df["kmeans_cluster"].unique()):
        count = (cluster_sorted == cl).sum()
        tick_positions.append(start + count / 2)
        tick_labels.append(f"Cluster {cl}{' (kidney)' if cl == target else ''}\n(n={count})")
        start += count
    ax.set_xticks(tick_positions); ax.set_xticklabels(tick_labels, fontsize=8.5)
    ax.set_yticks(tick_positions); ax.set_yticklabels(tick_labels, fontsize=8.5)
    ax.set_title(f"{slug}: pairwise distance matrix, sorted by KMeans cluster\n(darker = closer together)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "distance_heatmap.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[{slug}] saved distance_heatmap.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--group", required=True, choices=list(C.GROUPS))
    args = p.parse_args()
    main(args.group)
