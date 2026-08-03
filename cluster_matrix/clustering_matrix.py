"""
Clusters one study_group stratum from a participant-participant similarity
(adjacency) matrix instead of running KMeans directly on the feature space.

Method (spectral clustering, KMeans as the final assignment step):
  1. log1p the 3 skewed labs, StandardScaler the 5-feature panel -> X_scaled.
  2. Build a k-NN graph adjacency matrix on X_scaled: connect each point to
     its N_NEIGHBORS nearest neighbors (union-symmetrized), weight kept edges
     by the RBF kernel exp(-d^2 / (2*sigma^2)) with sigma = median kept-edge
     distance, all non-edges = 0. A *dense* RBF affinity (every pair weighted,
     no zeros) was tried first and failed: a global bandwidth lets 2-3 extreme
     outliers (e.g. diabetic-range HbA1c in the nominally healthy/prediabetic
     strata) end up with near-zero affinity to everyone, so isolating that
     handful of points trivially maximizes silhouette and produces a
     3-vs-756-style "cluster" that isn't a phenotype split. Sparsifying to a
     k-NN graph means no single point's global distances can dominate the cut.
     n_neighbors is bumped up automatically if the graph starts out
     disconnected (see connected_neighbor_graph below).
  3. Normalized graph Laplacian L_sym = I - deg^-1/2 A deg^-1/2.
  4. For each k in common.K_RANGE: take the k eigenvectors of L_sym with the
     smallest eigenvalues, row-normalize them (standard spectral-embedding
     step), and run KMeans on that embedding. k is chosen by silhouette
     score -- evaluated in the original standardized feature space (X_scaled)
     so it's comparable to the plain-KMeans fit in clusters_hba1c_by_group,
     not in the arbitrary eigenvector embedding.
  5. Refit at the chosen k and canonicalize cluster label order by mean
     standardized BMI+Insulin+C-peptide (adiposity/insulin-resistance axis),
     same convention as clusters_hba1c_by_group/clustering.py.

Before any of that, participants with |z| > common.OUTLIER_Z_THRESH on any one
of the 5 features (z computed from a preliminary StandardScaler fit on the
whole group) are dropped and logged to outliers.csv -- see
flag_and_remove_outliers. This is what actually fixes the "3 diabetic-range
HbA1c outliers in the healthy stratum get isolated as their own cluster"
problem; the k-NN graph alone only fixed it for 3 of the 4 study groups; the
healthy stratum's real weak point in the graph *was* those 3 outliers, not a
global-bandwidth artifact, so it had to be handled by removal, not sparsification.

Outputs (in cluster_matrix/<group>/):
  cluster_assignments.csv -- participant id, cluster, pc1-3 (outliers excluded)
  cluster_profiles.csv    -- per-cluster mean of each raw feature
  outliers.csv            -- dropped participants: raw values, z-scores, which feature(s) triggered removal
  similarity_matrix.png   -- k-NN graph adjacency matrix, rows/cols sorted by cluster
  clusters_3d.png         -- 3D PCA scatter (standardized feature space) colored by cluster
  meta.json               -- n, n_outliers_removed, best_k, silhouette scores, sigma, n_neighbors, PCA variance explained
"""
import argparse
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

N_NEIGHBORS_START = 15

# Load by absolute path under a directory-qualified module name rather than a
# bare `import common` -- other cluster_* directories each have their own
# unrelated common.py, and if one of those gets imported first in the same
# process, a bare `import common` would silently return the wrong module.
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


def connected_neighbor_graph(X_scaled, n_neighbors_start=N_NEIGHBORS_START):
    """k-NN graph adjacency (union-symmetrized, RBF-weighted edges), bumping
    n_neighbors up until the graph is a single connected component -- a
    disconnected graph gives a degenerate (non-unique) Laplacian null space."""
    n = len(X_scaled)
    n_neighbors = n_neighbors_start
    while True:
        conn = kneighbors_graph(X_scaled, n_neighbors=min(n_neighbors, n - 1),
                                 mode="distance", metric="euclidean", include_self=False)
        conn = conn.maximum(conn.T)  # union: keep edge if either point has the other as a kNN
        n_components, _ = connected_components(conn, directed=False)
        if n_components == 1 or n_neighbors >= n - 1:
            break
        n_neighbors += 5
    dist_dense = conn.toarray()
    edge_mask = dist_dense > 0
    sigma = np.median(dist_dense[edge_mask])
    affinity = np.zeros((n, n))
    affinity[edge_mask] = np.exp(-(dist_dense[edge_mask] ** 2) / (2 * sigma ** 2))
    return affinity, sigma, n_neighbors, int(n_components)


def flag_and_remove_outliers(df, X, idc):
    """Drop participants with |z| > common.OUTLIER_Z_THRESH on any one of the
    5 features (z from a StandardScaler fit on the whole group, i.e. before
    the outliers themselves are removed -- this is only used to *find* them).
    Returns (df_clean, X_clean, outliers_df) with df/X re-indexed 0..n-1."""
    z = StandardScaler().fit_transform(X)
    max_abs_z = np.max(np.abs(z), axis=1)
    is_outlier = max_abs_z > C.OUTLIER_Z_THRESH

    rows = []
    for i in np.where(is_outlier)[0]:
        triggers = []
        row = {"participant_id": df.loc[i, idc]}
        for j, c in enumerate(C.FEATURE_COLS):
            label = C.VAR_LABELS[c]
            row[f"{label}_raw"] = df.loc[i, c]
            row[f"{label}_z"] = round(float(z[i, j]), 2)
            if abs(z[i, j]) > C.OUTLIER_Z_THRESH:
                triggers.append(f"{label} (z={z[i, j]:+.1f})")
        row["trigger_features"] = "; ".join(triggers)
        rows.append(row)
    outliers_df = pd.DataFrame(rows)

    df_clean = df.loc[~is_outlier].reset_index(drop=True)
    X_clean = X.loc[~is_outlier].reset_index(drop=True)
    return df_clean, X_clean, outliers_df


def spectral_embedding(affinity, k):
    """Row-normalized top-k eigenvectors of the normalized graph Laplacian."""
    deg = affinity.sum(axis=1)
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(deg, 1e-12, None)))
    laplacian = np.eye(len(affinity)) - deg_inv_sqrt @ affinity @ deg_inv_sqrt
    # smallest k eigenvalues/vectors of a symmetric matrix
    eigvals, eigvecs = eigh(laplacian, subset_by_index=[0, k - 1])
    row_norms = np.linalg.norm(eigvecs, axis=1, keepdims=True)
    return eigvecs / np.clip(row_norms, 1e-12, None)


def main(slug):
    OUT_DIR = C.out_dir(slug)
    df = C.load_raw()
    df = C.filter_group(df, slug)
    df = df.dropna(subset=C.FEATURE_COLS).reset_index(drop=True)
    print(f"[{slug}] n={len(df)} (complete cases on {C.FEATURE_COLS})")

    X = df[C.FEATURE_COLS].copy()
    for c in C.LOG_COLS:
        X[c] = np.log1p(X[c])

    idc = C.id_col(df)
    df, X, outliers_df = flag_and_remove_outliers(df, X, idc)
    outliers_df.to_csv(os.path.join(OUT_DIR, "outliers.csv"), index=False)
    if len(outliers_df):
        print(f"[{slug}] removed {len(outliers_df)} outlier(s) (|z| > {C.OUTLIER_Z_THRESH}), n={len(df)} remain:")
        for _, r in outliers_df.iterrows():
            print(f"    id {r['participant_id']}: {r['trigger_features']}")
    else:
        print(f"[{slug}] no outliers above |z| > {C.OUTLIER_Z_THRESH}")

    # Re-standardize on the outlier-free group -- so the removed points can't
    # skew the mean/SD used for everyone who's actually being clustered.
    X_scaled = StandardScaler().fit_transform(X)

    affinity, sigma, n_neighbors, n_components = connected_neighbor_graph(X_scaled)
    print(f"[{slug}] k-NN graph: n_neighbors={n_neighbors}, sigma (median kept-edge dist) = {sigma:.4f}, "
          f"connected_components={n_components}")

    silhouettes = []
    for k in C.K_RANGE:
        emb = spectral_embedding(affinity, k)
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(emb)
        silhouettes.append(silhouette_score(X_scaled, labels))
    best_k = list(C.K_RANGE)[int(np.argmax(silhouettes))]
    print(f"[{slug}] silhouette by k: {dict(zip(C.K_RANGE, [round(s, 3) for s in silhouettes]))}")
    print(f"[{slug}] chosen k = {best_k}")

    emb = spectral_embedding(affinity, best_k)
    raw_labels = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit_predict(emb)

    # Canonicalize cluster labels the same way as clusters_hba1c_by_group:
    # order by mean standardized BMI+Insulin+C-peptide ascending, so cluster 0
    # is always the low-adiposity/low-insulin group -- comparable across the
    # 4 study groups and against the plain-KMeans fit.
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

    # --- similarity/adjacency matrix heatmap, sorted by cluster ---
    order = df.sort_values("cluster").index.to_numpy()
    affinity_sorted = affinity[np.ix_(order, order)]
    cluster_sorted = df.loc[order, "cluster"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.5, 7.5), facecolor="#fcfcfb")
    im = ax.imshow(affinity_sorted, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"RBF-weighted edge (k-NN graph, k={n_neighbors}); 0 = not connected", fontsize=10)
    boundaries = np.where(np.diff(cluster_sorted) != 0)[0] + 1
    for b in boundaries:
        ax.axhline(b, color="white", linewidth=1.2)
        ax.axvline(b, color="white", linewidth=1.2)
    tick_positions, tick_labels = [], []
    start = 0
    for cl in range(best_k):
        count = int((cluster_sorted == cl).sum())
        tick_positions.append(start + count / 2)
        tick_labels.append(f"Cluster {cl}\n(n={count})")
        start += count
    ax.set_xticks(tick_positions); ax.set_xticklabels(tick_labels, fontsize=8.5)
    ax.set_yticks(tick_positions); ax.set_yticklabels(tick_labels, fontsize=8.5)
    ax.set_title(f"{slug}: k-NN graph adjacency matrix, sorted by spectral+KMeans cluster\n(brighter = more similar; black = no edge)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "similarity_matrix.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)

    # --- 3D scatter of the clusters in standardized-feature PCA space ---
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
    ax.set_title(f"{slug}: spectral affinity + KMeans (k={best_k}) on age/BMI/insulin/C-peptide/HbA1c, n={len(df)}", fontsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "clusters_3d.png"), dpi=200, facecolor=fig.get_facecolor())
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

    df[[idc, "cluster", "pc1", "pc2", "pc3"]].to_csv(os.path.join(OUT_DIR, "cluster_assignments.csv"), index=False)

    C.save_meta(
        slug,
        study_group=C.GROUPS[slug],
        n=len(df),
        n_outliers_removed=len(outliers_df),
        outlier_ids=outliers_df["participant_id"].tolist() if len(outliers_df) else [],
        outlier_z_thresh=C.OUTLIER_Z_THRESH,
        method="spectral (k-NN graph adjacency, RBF-weighted edges + normalized Laplacian + KMeans on top-k eigenvectors)",
        n_neighbors=n_neighbors,
        rbf_sigma=round(float(sigma), 4),
        best_k=best_k,
        silhouette_by_k={int(k): round(float(s), 4) for k, s in zip(C.K_RANGE, silhouettes)},
        pca_variance_explained=[round(float(v), 4) for v in var_explained],
    )
    print(f"[{slug}] saved cluster_assignments.csv, cluster_profiles.csv, outliers.csv, "
          "similarity_matrix.png, clusters_3d.png, meta.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--group", required=True, choices=list(C.GROUPS))
    args = p.parse_args()
    main(args.group)
