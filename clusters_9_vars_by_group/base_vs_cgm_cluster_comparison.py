"""
Stage 1d, stratified: does adding CGM features to the within-group clustering
reshuffle membership, or agree with the base (no-CGM) clustering? Uses the
auto-identified kidney-complication cluster (meta.json: target_cluster_kmeans)
rather than a hardcoded cluster index.

Output: base_vs_cgm_crosstab.csv, base_vs_cgm_crosstab.png
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main(slug):
    OUT_DIR = C.out_dir(slug)
    meta = C.load_meta(slug)
    target = meta["target_cluster_kmeans"]

    base = pd.read_csv(os.path.join(OUT_DIR, "clustering_base_assignments.csv"))
    cgm = pd.read_csv(os.path.join(OUT_DIR, "clustering_with_cgm_assignments.csv"))

    idc = base.columns[0]
    merged = base[[idc, "kmeans_cluster"]].merge(cgm[[idc, "kmeans_cluster"]], on=idc, suffixes=("_base", "_cgm"))
    print(f"[{slug}] n={len(merged)} participants present in both clusterings")

    ari = adjusted_rand_score(merged["kmeans_cluster_base"], merged["kmeans_cluster_cgm"])
    print(f"[{slug}] Adjusted Rand Index (base vs with-CGM) = {ari:.3f}")

    crosstab = pd.crosstab(merged["kmeans_cluster_base"], merged["kmeans_cluster_cgm"])
    print(f"[{slug}] Base cluster (rows) vs with-CGM cluster (cols) crosstab:")
    print(crosstab)
    crosstab.to_csv(os.path.join(OUT_DIR, "base_vs_cgm_crosstab.csv"))

    target_base = merged[merged["kmeans_cluster_base"] == target]
    print(f"[{slug}] Base kidney-complication cluster {target} (n={len(target_base)}) -> with-CGM cluster distribution:")
    print(target_base["kmeans_cluster_cgm"].value_counts().sort_index())

    fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.imshow(crosstab.values, cmap="Blues")
    for i in range(crosstab.shape[0]):
        for j in range(crosstab.shape[1]):
            ax.text(j, i, str(crosstab.values[i, j]), ha="center", va="center",
                     color="white" if crosstab.values[i, j] > crosstab.values.max() / 2 else "black")
    ax.set_xticks(range(crosstab.shape[1])); ax.set_xticklabels([f"with-CGM {c}" for c in crosstab.columns])
    row_labels = [f"Base {c}" + (" (kidney)" if c == target else "") for c in crosstab.index]
    ax.set_yticks(range(crosstab.shape[0])); ax.set_yticklabels(row_labels)
    ax.set_title(f"{slug}: base vs with-CGM clusters (ARI={ari:.3f})", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "base_vs_cgm_crosstab.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[{slug}] saved base_vs_cgm_crosstab.csv, base_vs_cgm_crosstab.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--group", required=True, choices=list(C.GROUPS))
    args = p.parse_args()
    main(args.group)
