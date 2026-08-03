"""
Clusters one study_group stratum on a 5-variable panel: age, BMI, insulin,
C-peptide, HbA1c. log1p on the 3 skewed labs, StandardScaler, KMeans with k
chosen by silhouette over common.K_RANGE.

Outputs (in clusters_hba1c_by_group/<group>/):
  cluster_assignments.csv -- participant id, cluster, pc1-3
  cluster_profiles.csv    -- per-cluster mean of each raw feature
  clusters_3d.png         -- 3D PCA scatter colored by cluster
  cluster_violins.png     -- one violin panel per feature, split by cluster
  meta.json               -- n, best_k, silhouette scores, PCA variance explained
"""
import argparse
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Load by absolute path under a directory-qualified module name rather than a
# bare `import common` -- clusters_9_vars_by_group/ has its own unrelated
# common.py, and if that gets imported first in the same process, a bare
# `import common` silently returns the wrong module instead of raising. See
# the matching comment in avg_linkage_distance.py.
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
    raw_labels = kmeans.labels_

    # Canonicalize cluster labels: order by mean standardized BMI+Insulin+C-peptide
    # (the adiposity/insulin-resistance axis PC1 loads on) ascending, so cluster 0
    # is always the low-adiposity/low-insulin group and higher indices are always
    # higher -- comparable across the 4 study groups. Raw KMeans label indices are
    # otherwise arbitrary per fit (e.g. they came out flipped for oral_medication
    # before this fix).
    composite_idx = [C.FEATURE_COLS.index(c) for c in (C.BMI_COL, C.INSULIN_COL, C.CPEPTIDE_COL)]
    composite_score = X_scaled[:, composite_idx].mean(axis=1)
    cluster_order = pd.Series(composite_score).groupby(raw_labels).mean().sort_values().index.tolist()
    relabel_map = {old: new for new, old in enumerate(cluster_order)}
    df["cluster"] = pd.Series(raw_labels).map(relabel_map).values

    pca = PCA(n_components=3, random_state=0)
    X_pca = pca.fit_transform(X_scaled)
    df["pc1"], df["pc2"], df["pc3"] = X_pca[:, 0], X_pca[:, 1], X_pca[:, 2]
    var_explained = pca.explained_variance_ratio_
    print(f"[{slug}] PCA variance explained: PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%}, "
          f"PC3={var_explained[2]:.1%} (cumulative {var_explained.sum():.1%})")

    loadings = pd.DataFrame(pca.components_.T, index=[C.VAR_LABELS[c] for c in C.FEATURE_COLS], columns=["PC1", "PC2", "PC3"])
    loadings.to_csv(os.path.join(OUT_DIR, "pca_loadings.csv"))

    # --- 3D scatter of the clusters in PCA space ---
    fig = plt.figure(figsize=(7.5, 6.5), facecolor="#fcfcfb")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#fcfcfb")
    ax.set_xlabel(f"PC1 ({var_explained[0]:.0%})", fontsize=9)
    ax.set_ylabel(f"PC2 ({var_explained[1]:.0%})", fontsize=9)
    ax.set_zlabel(f"PC3 ({var_explained[2]:.0%})", fontsize=9)
    ax.view_init(elev=C.VIEW_ELEV, azim=C.VIEW_AZIM)
    for cl in range(best_k):
        sub = df[df["cluster"] == cl]
        ax.scatter(
            sub["pc1"], sub["pc2"], sub["pc3"], s=10, alpha=0.55,
            color=C.CLUSTER_PALETTE[cl % len(C.CLUSTER_PALETTE)],
            label=f"Cluster {cl} (n={len(sub)})",
        )
    ax.set_title(f"{slug}: KMeans (k={best_k}) on age/BMI/insulin/C-peptide/HbA1c, n={len(df)}", fontsize=10.5)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "clusters_3d.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)

    # --- Violin plots: one panel per cluster, all 5 features side by side ---
    # Features are z-scored (same standardization used for clustering) since raw
    # units span wildly different ranges (age ~40-90 vs insulin ~0-12) and can't
    # share one y-axis; z-units make each cluster's shape across variables legible.
    zdf = pd.DataFrame(X_scaled, columns=C.FEATURE_COLS)
    zdf["cluster"] = df["cluster"].values
    feature_labels = [C.VAR_LABELS[c] for c in C.FEATURE_COLS]

    fig, axes = plt.subplots(1, best_k, figsize=(3.3 * best_k, 4.8), facecolor="#fcfcfb", sharey=True)
    if best_k == 1:
        axes = [axes]
    for cl, ax_i in zip(range(best_k), axes):
        ax_i.set_facecolor("#fcfcfb")
        sub = zdf[zdf["cluster"] == cl]
        data_by_feature = [sub[c].values for c in C.FEATURE_COLS]
        color = C.CLUSTER_PALETTE[cl % len(C.CLUSTER_PALETTE)]
        parts = ax_i.violinplot(data_by_feature, showmeans=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.65)
        parts["cmeans"].set_color("#33322e")
        ax_i.axhline(0, color="#c9c7c0", linewidth=1, zorder=0)
        ax_i.set_xticks(range(1, len(C.FEATURE_COLS) + 1))
        ax_i.set_xticklabels(feature_labels, fontsize=8.5, rotation=20, ha="right")
        ax_i.set_title(f"Cluster {cl} (n={len(sub)})", fontsize=10.5, color=color)
        ax_i.spines["top"].set_visible(False)
        ax_i.spines["right"].set_visible(False)
    axes[0].set_ylabel("standardized value (z-score)", fontsize=9)
    fig.suptitle(f"{slug}: cluster profiles across features (n={len(df)})", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "cluster_violins.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)

    profiles = []
    for cl in range(best_k):
        sub = df[df["cluster"] == cl]
        row = {"cluster": cl, "n": len(sub)}
        for c in C.FEATURE_COLS:
            row[C.VAR_LABELS[c]] = sub[c].mean()
        profiles.append(row)
    profiles_df = pd.DataFrame(profiles)
    profiles_df.to_csv(os.path.join(OUT_DIR, "cluster_profiles.csv"), index=False)
    print(f"[{slug}] cluster profiles:")
    print(profiles_df.to_string(index=False))

    idc = C.id_col(df)
    df[[idc, "cluster", "pc1", "pc2", "pc3"]].to_csv(os.path.join(OUT_DIR, "cluster_assignments.csv"), index=False)

    C.save_meta(
        slug,
        study_group=C.GROUPS[slug],
        n=len(df),
        best_k=best_k,
        silhouette_by_k={int(k): round(float(s), 4) for k, s in zip(C.K_RANGE, silhouettes)},
        pca_variance_explained=[round(float(v), 4) for v in var_explained],
    )
    print(f"[{slug}] saved cluster_assignments.csv, cluster_profiles.csv, pca_loadings.csv, "
          "clusters_3d.png, cluster_violins.png, meta.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--group", required=True, choices=list(C.GROUPS))
    args = p.parse_args()
    main(args.group)
