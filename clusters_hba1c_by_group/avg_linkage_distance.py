"""
Average-linkage distance between clusters, per study_group stratum -- plus
average intra-cluster distance, since a between-cluster distance is only
meaningful relative to how spread out each cluster is on its own.

Average linkage = mean pairwise Euclidean distance between all points in
cluster A and all points in cluster B. Intra-cluster distance = mean pairwise
Euclidean distance between all points within a single cluster. Both computed
in the same standardized (log1p + z-score) feature space used for the KMeans
fit in clustering.py. Average linkage is NOT the centroid distance -- it
accounts for within-cluster spread, not just cluster means.

Outputs (in clusters_hba1c_by_group/):
  avg_linkage_distances.csv -- group, cluster_a, cluster_b, n_a, n_b,
                                intra_dist_a, intra_dist_b, avg_linkage_distance
  avg_linkage_distances.png -- table, one row per group (all groups have k=2)
"""
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from sklearn.preprocessing import StandardScaler

# Load this directory's common.py by absolute path under a directory-qualified
# module name, rather than a bare `import common`. clusters_9_vars_by_group/
# has its own unrelated common.py -- if that gets imported first in the same
# process (e.g. both by-group pipelines run in one notebook kernel), a bare
# `import common` silently returns THAT module (no HBA1C_COL, wrong OUT_ROOT)
# instead of raising, and compute_group() fails looking for
# cluster_assignments.csv under the wrong directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_COMMON_KEY = f"common__{os.path.basename(_THIS_DIR)}"
if _COMMON_KEY in sys.modules:
    C = sys.modules[_COMMON_KEY]
else:
    _spec = importlib.util.spec_from_file_location(_COMMON_KEY, os.path.join(_THIS_DIR, "common.py"))
    C = importlib.util.module_from_spec(_spec)
    sys.modules[_COMMON_KEY] = C
    _spec.loader.exec_module(C)

GROUP_LABELS = {
    "healthy": "Healthy",
    "prediabetic": "Prediabetic",
    "oral_medication": "Oral medication",
    "insulin_dependent": "Insulin dependent",
}


def compute_group(slug):
    df = C.load_raw()
    df = C.filter_group(df, slug)
    df = df.dropna(subset=C.FEATURE_COLS).reset_index(drop=True)

    X = df[C.FEATURE_COLS].copy()
    for c in C.LOG_COLS:
        X[c] = np.log1p(X[c])
    X_scaled = StandardScaler().fit_transform(X)

    idc = C.id_col(df)
    assign = pd.read_csv(os.path.join(C.out_dir(slug), "cluster_assignments.csv"))
    merged = df[[idc]].merge(assign[[idc, "cluster"]], on=idc, how="left")
    assert merged["cluster"].notna().all(), f"[{slug}] unmatched ids between raw data and cluster_assignments.csv"
    clusters = merged["cluster"].astype(int).values

    labels = sorted(set(clusters))
    intra = {}
    for lbl in labels:
        Xl = X_scaled[clusters == lbl]
        intra[lbl] = float(pdist(Xl, metric="euclidean").mean()) if len(Xl) > 1 else float("nan")

    rows = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            Xa = X_scaled[clusters == a]
            Xb = X_scaled[clusters == b]
            d = cdist(Xa, Xb, metric="euclidean").mean()
            rows.append({
                "group": slug, "cluster_a": a, "cluster_b": b,
                "n_a": len(Xa), "n_b": len(Xb),
                "intra_dist_a": round(intra[a], 4),
                "intra_dist_b": round(intra[b], 4),
                "avg_linkage_distance": round(float(d), 4),
            })
    return rows


def table(out):
    # All 4 groups fit k=2 -> one row per group. If a future rerun ever
    # produces k>2 for some group, each cluster pair gets its own row instead.
    rows = out.copy()
    rows["cluster pair"] = "cl " + rows["cluster_a"].astype(str) + "–" + rows["cluster_b"].astype(str)
    rows["Study group"] = rows["group"].map(GROUP_LABELS)
    rows["separation ratio"] = rows["avg_linkage_distance"] / rows[["intra_dist_a", "intra_dist_b"]].mean(axis=1)
    rows = rows[["Study group", "cluster pair", "n_a", "intra_dist_a", "n_b", "intra_dist_b",
                 "avg_linkage_distance", "separation ratio"]].rename(columns={
        "n_a": "n (cluster a)", "intra_dist_a": "intra-cluster dist (a)",
        "n_b": "n (cluster b)", "intra_dist_b": "intra-cluster dist (b)",
        "avg_linkage_distance": "Between-cluster avg. linkage",
    })
    col_order = list(GROUP_LABELS.values())
    rows["Study group"] = pd.Categorical(rows["Study group"], categories=col_order, ordered=True)
    rows = rows.sort_values("Study group").reset_index(drop=True)

    col_labels = list(rows.columns)
    cell_text = [
        [r["Study group"], r["cluster pair"], str(r["n (cluster a)"]), f"{r['intra-cluster dist (a)']:.2f}",
         str(r["n (cluster b)"]), f"{r['intra-cluster dist (b)']:.2f}",
         f"{r['Between-cluster avg. linkage']:.2f}", f"{r['separation ratio']:.2f}x"]
        for _, r in rows.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(11, 0.55 * (len(rows) + 1) + 0.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.axis("off")
    ax.set_title(
        "Between-cluster vs. intra-cluster distance, by study group\n"
        "mean pairwise Euclidean distance, 5-feature z-scored space; separation ratio = "
        "between-cluster / mean intra-cluster",
        fontsize=10.5, color="#0b0b0b", pad=14,
    )

    tbl = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.8)
    tbl.auto_set_column_width(col=list(range(len(col_labels))))

    n_cols = len(col_labels)
    for (row_i, col_i), cell in tbl.get_celld().items():
        cell.set_edgecolor("#e1e0d9")
        cell.set_linewidth(1)
        if row_i == 0:
            cell.set_facecolor("#f0efec")
            cell.set_text_props(color="#52514e", fontweight="bold")
        else:
            cell.set_facecolor("#fcfcfb" if row_i % 2 else "#f9f9f7")
            cell.set_text_props(color="#0b0b0b")
            if col_i >= n_cols - 2:
                cell.set_text_props(color="#0b0b0b", fontweight="bold", fontfamily="monospace")

    fig.tight_layout()
    out_path = os.path.join(C.OUT_ROOT, "avg_linkage_distances.png")
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    all_rows = []
    for slug in C.GROUPS:
        all_rows.extend(compute_group(slug))
    out = pd.DataFrame(all_rows)
    out_path = os.path.join(C.OUT_ROOT, "avg_linkage_distances.csv")
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"saved {out_path}")
    table(out)


if __name__ == "__main__":
    main()
