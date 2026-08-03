"""
Violin plots per study_group, organized by variable rather than by cluster:
one subplot per feature, x-axis = cluster (from clustering_matrix.py's
spectral+KMeans fit), each violin showing that variable's distribution
within the cluster.

Unlike clusters_hba1c_by_group's cluster_violins.png (one panel per cluster,
all features z-scored onto a shared y-axis so they can share one scale), here
each subplot keeps its own natural, untransformed scale: no log1p and no
standardization, so units are each feature's raw units and not comparable
across subplots.

Outputs (in cluster_matrix/<group>/):
  violins_by_variable.png
"""
import argparse
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
_COMMON_KEY = f"common__{os.path.basename(_THIS_DIR)}"
if _COMMON_KEY in sys.modules:
    C = sys.modules[_COMMON_KEY]
else:
    _spec = importlib.util.spec_from_file_location(_COMMON_KEY, os.path.join(_THIS_DIR, "common.py"))
    C = importlib.util.module_from_spec(_spec)
    sys.modules[_COMMON_KEY] = C
    _spec.loader.exec_module(C)


def main(slug):
    OUT_DIR = C.out_dir(slug)
    df = C.load_raw()
    df = C.filter_group(df, slug)
    df = df.dropna(subset=C.FEATURE_COLS).reset_index(drop=True)

    idc = C.id_col(df)
    assign = pd.read_csv(os.path.join(OUT_DIR, "cluster_assignments.csv"))
    n_before = len(df)
    df = df.merge(assign[[idc, "cluster"]], on=idc, how="inner")
    # rows that don't match were dropped as outliers by clustering_matrix.py
    # (see outliers.csv in this same directory), not a data problem here.
    print(f"[{slug}] {n_before - len(df)} participant(s) excluded as pre-identified outliers (see outliers.csv)")
    df["cluster"] = df["cluster"].astype(int)
    best_k = int(df["cluster"].max()) + 1

    # No log1p, no standardization -- each subplot is on its own natural raw scale.
    plot_df = df[C.FEATURE_COLS + ["cluster"]].copy()

    n_features = len(C.FEATURE_COLS)
    fig, axes = plt.subplots(1, n_features, figsize=(3.6 * n_features, 4.8), facecolor="#fcfcfb")
    if n_features == 1:
        axes = [axes]

    for feat, ax in zip(C.FEATURE_COLS, axes):
        ax.set_facecolor("#fcfcfb")
        data_by_cluster = [plot_df.loc[plot_df["cluster"] == cl, feat].values for cl in range(best_k)]
        parts = ax.violinplot(data_by_cluster, showmeans=True, showextrema=False)
        for cl, body in enumerate(parts["bodies"]):
            color = C.CLUSTER_PALETTE[cl % len(C.CLUSTER_PALETTE)]
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.65)
        parts["cmeans"].set_color("#33322e")
        ax.set_xticks(range(1, best_k + 1))
        ax.set_xticklabels([f"Cluster {cl}\n(n={(plot_df['cluster'] == cl).sum()})" for cl in range(best_k)],
                            fontsize=8.5)
        ax.set_ylabel(C.VAR_LABELS[feat], fontsize=9.5)
        ax.set_title(C.VAR_LABELS[feat], fontsize=10.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"{slug}: variable distributions by cluster (n={len(df)}, k={best_k}, own scale per variable)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "violins_by_variable.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[{slug}] saved violins_by_variable.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--group", required=True, choices=list(C.GROUPS))
    args = p.parse_args()
    main(args.group)
