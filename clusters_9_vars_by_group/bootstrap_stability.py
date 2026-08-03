"""
Stage 1a, stratified: bootstrap stability of the within-group 9-variable
KMeans clustering (k = meta.json's best_k for this group). Resample with
replacement 100x, refit KMeans each time, compute per-original-cluster
co-clustering stability. Matters most for the small kidney-complication
cluster, especially in the smaller strata (insulin_dependent).

Output: bootstrap_stability.png
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

N_BOOT = 100


def main(slug):
    OUT_DIR = C.out_dir(slug)
    meta = C.load_meta(slug)
    best_k = meta["best_k"]
    target = meta["target_cluster_kmeans"]

    df = C.load_raw()
    df = C.filter_group(df, slug)
    df = df.dropna(subset=C.FEATURE_COLS).reset_index(drop=True)
    n = len(df)
    print(f"[{slug}] n={n}, k={best_k}")

    X = df[C.FEATURE_COLS].copy()
    for c in C.LOG_COLS:
        X[c] = np.log1p(X[c])
    X_scaled = StandardScaler().fit_transform(X)

    original_labels = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit(X_scaled).labels_
    cluster_ids = sorted(set(original_labels))
    print(f"[{slug}] original cluster sizes: {[(c, int((original_labels == c).sum())) for c in cluster_ids]}")

    together_count = np.zeros((n, n), dtype=np.int32)
    seen_count = np.zeros((n, n), dtype=np.int32)

    rng = np.random.default_rng(0)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        X_boot = X_scaled[idx]
        labels_boot = KMeans(n_clusters=best_k, n_init=10, random_state=b).fit(X_boot).labels_

        unique_idx, first_pos = np.unique(idx, return_index=True)
        labels_for_unique = labels_boot[first_pos]

        seen_count[np.ix_(unique_idx, unique_idx)] += 1
        same = labels_for_unique[:, None] == labels_for_unique[None, :]
        together_count[np.ix_(unique_idx, unique_idx)] += same

    with np.errstate(invalid="ignore", divide="ignore"):
        stability_matrix = np.where(seen_count > 0, together_count / np.maximum(seen_count, 1), np.nan)

    stability_per_cluster = {}
    for c in cluster_ids:
        members = np.where(original_labels == c)[0]
        pairs = stability_matrix[np.ix_(members, members)]
        iu = np.triu_indices(len(members), k=1)
        stability_per_cluster[c] = np.nanmean(pairs[iu]) if len(members) > 1 else np.nan
        print(f"[{slug}] Cluster {c} (n={len(members)}){' [kidney]' if c == target else ''}: "
              f"mean pairwise co-clustering stability = {stability_per_cluster[c]:.3f}")

    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    sizes = [int((original_labels == c).sum()) for c in cluster_ids]
    labels = [f"Cluster {c}{' (kidney)' if c == target else ''}\n(n={s})" for c, s in zip(cluster_ids, sizes)]
    bars = ax.bar(labels, [stability_per_cluster[c] for c in cluster_ids],
                  color=[C.CLUSTER_PALETTE[c % len(C.CLUSTER_PALETTE)] for c in cluster_ids])
    ax.axhline(1.0, color="#999999", linewidth=1.0, linestyle="--", zorder=1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean pairwise co-clustering stability\n(fraction of 100 bootstrap resamples)", fontsize=10.5)
    ax.set_title(f"{slug}: bootstrap stability of 9-var KMeans clusters (k={best_k}, {N_BOOT} resamples)", fontsize=11.5)
    for bar, c in zip(bars, cluster_ids):
        val = stability_per_cluster[c]
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", fontsize=10)
    ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "bootstrap_stability.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[{slug}] saved bootstrap_stability.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--group", required=True, choices=list(C.GROUPS))
    args = p.parse_args()
    main(args.group)
