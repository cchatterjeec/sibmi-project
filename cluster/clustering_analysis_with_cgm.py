"""
KMeans and hierarchical (Ward) clustering, extending clustering_analysis.py's
5 variables (age, HbA1c, BMI, C-peptide, insulin) with the 21-feature
VIF-pruned/curated CGM set (regression_no_correlated_features/
select_cgm_features_vif.py) -- the curated set is used rather than the
full 66-feature catalog for the same reason it was built in the first
place: clustering is distance-based, so severely collinear CGM features
would silently dominate the distance metric.

Unlike clustering_analysis.py (which uses the full cohort with no CGM
gate, since none of its 5 variables are CGM-derived), this script DOES
require the standard qc_pct_active >= 70% CGM quality filter, since CGM
features are now part of the feature set and unreliable CGM data would
corrupt the clustering directly.

This is a separate script from clustering_analysis.py -- it does not
overwrite or replace it or its outputs.

Insulin and C-peptide are still log1p-transformed before standardizing
(right-skewed lab concentrations); the 21 CGM features are z-scored as-is,
consistent with how they're handled everywhere else in this project (no
additional log transform).

Outputs (in this directory):
  with_cgm_elbow_silhouette.png   -- k selection: inertia (elbow) + silhouette vs k
  with_cgm_dendrogram.png         -- truncated Ward dendrogram
  with_cgm_pca_clusters.png       -- 4-panel PCA scatter: KMeans / Hierarchical /
                                      study_group / HbA1c group
  with_cgm_pca_loadings.png/.csv  -- top variable loadings on PC1/PC2
  with_cgm_cluster_profiles.csv   -- per-cluster mean of each raw variable, n,
                                      study_group composition
  with_cgm_cluster_assignments.csv -- participant_id, KMeans cluster,
                                      Hierarchical cluster, study_group, hba1c_group
"""
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(OUT_DIR)
DATA_PATH = os.path.join(ROOT, "final_df.csv")
K_RANGE = range(2, 9)
CGM_QC_MIN_PCT_ACTIVE = 70.0

sys.path.insert(0, os.path.join(ROOT, "regression_no_correlated_features"))
from select_cgm_features_vif import CGM_FEATURES_VIF_PRUNED


def sanitize_column_name(col):
    return re.sub(r"[\[\]<]", "", col)


HBA1C_COL = sanitize_column_name("import_hba1c, Hemoglobin A1c/Hemoglobin.total in ")
CPEPTIDE_COL = sanitize_column_name("import_c_peptide, C peptide [Mass/volume] in Seru")
INSULIN_COL = sanitize_column_name("import_insulin, Insulin [Units/volume] in Serum o")
BMI_COL = "bmi_vsorres, BMI"

BASE_VAR_LABELS = {
    "age": "Age", HBA1C_COL: "HbA1c", BMI_COL: "BMI", CPEPTIDE_COL: "C-peptide", INSULIN_COL: "Insulin",
}
VAR_LABELS = {**BASE_VAR_LABELS, **{c: c for c in CGM_FEATURES_VIF_PRUNED}}
BASE_FEATURE_COLS = ["age", HBA1C_COL, BMI_COL, CPEPTIDE_COL, INSULIN_COL]
FEATURE_COLS = BASE_FEATURE_COLS + CGM_FEATURES_VIF_PRUNED
LOG_COLS = {CPEPTIDE_COL, INSULIN_COL}

CLUSTER_PALETTE = ["#2a78d6", "#eb6834", "#2ca858", "#a259c6", "#d6b02a", "#d64550"]


def main():
    df = pd.read_csv(DATA_PATH)
    df.columns = [sanitize_column_name(c) for c in df.columns]
    before = len(df)
    df = df[df["qc_pct_active"] >= CGM_QC_MIN_PCT_ACTIVE]
    print(f"[QC filter] dropped {before - len(df)} participants with qc_pct_active < {CGM_QC_MIN_PCT_ACTIVE}% or missing")
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    print(f"n={len(df)} (complete cases on {len(FEATURE_COLS)} features: "
          f"{len(BASE_FEATURE_COLS)} base + {len(CGM_FEATURES_VIF_PRUNED)} curated CGM)")

    X = df[FEATURE_COLS].copy()
    for c in LOG_COLS:
        X[c] = np.log1p(X[c])
    X_scaled = StandardScaler().fit_transform(X)

    # --- k selection: elbow (inertia) + silhouette ---
    inertias, silhouettes = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))

    best_k = list(K_RANGE)[int(np.argmax(silhouettes))]
    print(f"silhouette by k: {dict(zip(K_RANGE, [round(s, 3) for s in silhouettes]))}")
    print(f"chosen k (max silhouette) = {best_k}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.xaxis.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[0].plot(list(K_RANGE), inertias, color="#2a78d6", marker="o", markersize=6, linewidth=1.6)
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia"); axes[0].set_title("Elbow plot", fontsize=11)
    axes[1].plot(list(K_RANGE), silhouettes, color="#eb6834", marker="o", markersize=6, linewidth=1.6)
    axes[1].axvline(best_k, color="#999999", linewidth=1.0, linestyle="--", zorder=1)
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette score"); axes[1].set_title("Silhouette score", fontsize=11)
    fig.suptitle(f"KMeans k selection (5 base vars + {len(CGM_FEATURES_VIF_PRUNED)} curated CGM features)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "with_cgm_elbow_silhouette.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)

    # --- KMeans at chosen k ---
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit(X_scaled)
    df["kmeans_cluster"] = kmeans.labels_

    # --- Hierarchical (Ward) ---
    Z = linkage(X_scaled, method="ward")
    df["hier_cluster"] = fcluster(Z, t=best_k, criterion="maxclust") - 1

    ari = adjusted_rand_score(df["kmeans_cluster"], df["hier_cluster"])
    print(f"Adjusted Rand Index (KMeans vs Hierarchical agreement) = {ari:.3f}")

    fig, ax = plt.subplots(figsize=(11, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    dendrogram(Z, truncate_mode="lastp", p=40, show_leaf_counts=True, leaf_font_size=8, ax=ax,
               color_threshold=Z[-(best_k - 1), 2] if best_k > 1 else None)
    ax.set_title(f"Ward hierarchical clustering dendrogram (truncated, last 40 merges)\ncut at k={best_k} shown by color", fontsize=12)
    ax.set_xlabel("Cluster size (or sample index)")
    ax.set_ylabel("Ward distance")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "with_cgm_dendrogram.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)

    # --- PCA projection for visualization ---
    pca = PCA(n_components=2, random_state=0)
    X_pca = pca.fit_transform(X_scaled)
    df["pc1"], df["pc2"] = X_pca[:, 0], X_pca[:, 1]
    var_explained = pca.explained_variance_ratio_
    print(f"PCA variance explained: PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%}")

    loadings = pd.DataFrame(pca.components_.T, index=[VAR_LABELS[c] for c in FEATURE_COLS], columns=["PC1", "PC2"])
    loadings.to_csv(os.path.join(OUT_DIR, "with_cgm_pca_loadings.csv"))
    top_loadings = loadings.reindex(loadings.abs().sum(axis=1).sort_values(ascending=False).index).head(15)
    print("\nTop 15 PCA loadings by |PC1|+|PC2|:")
    print(top_loadings.round(3).to_string())

    fig_l, axes_l = plt.subplots(1, 2, figsize=(11, 6), facecolor="#fcfcfb")
    for i, pc in enumerate(["PC1", "PC2"]):
        ax = axes_l[i]
        ax.set_facecolor("#fcfcfb")
        vals = top_loadings[pc].sort_values()
        colors = ["#d64550" if v < 0 else "#2a78d6" for v in vals]
        ax.barh(vals.index, vals.values, color=colors)
        ax.axvline(0, color="#999999", linewidth=1.0)
        ax.set_title(f"{pc} loadings ({var_explained[i]:.0%} of variance)\n(top 15 variables by combined |PC1|+|PC2|)", fontsize=10.5)
        ax.set_xlabel("Weight on this component")
        ax.tick_params(axis="y", labelsize=8)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    fig_l.suptitle("What each variable contributes to PC1 / PC2 (5 base vars + 21 curated CGM)", fontsize=12)
    fig_l.tight_layout()
    fig_l.savefig(os.path.join(OUT_DIR, "with_cgm_pca_loadings.png"), dpi=200, facecolor=fig_l.get_facecolor())
    plt.close(fig_l)

    # ADA HbA1c categories, computed directly from the actual HbA1c value.
    hba1c_bins = [-np.inf, 5.7, 6.5, np.inf]
    hba1c_group_labels = ["Normal (<5.7%)", "Prediabetes (5.7-6.4%)", "Diabetes (>=6.5%)"]
    df["hba1c_group"] = pd.cut(df[HBA1C_COL], bins=hba1c_bins, labels=hba1c_group_labels, right=False)
    HBA1C_GROUP_COLORS = {
        "Normal (<5.7%)": "#2a78d6", "Prediabetes (5.7-6.4%)": "#d6b02a", "Diabetes (>=6.5%)": "#d64550",
    }

    study_groups = sorted(df["study_group"].dropna().unique().tolist())
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2), facecolor="#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.set_xlabel(f"PC1 ({var_explained[0]:.0%})")
        ax.set_ylabel(f"PC2 ({var_explained[1]:.0%})")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    for cl in range(best_k):
        sub = df[df["kmeans_cluster"] == cl]
        axes[0].scatter(sub["pc1"], sub["pc2"], s=10, alpha=0.6, color=CLUSTER_PALETTE[cl % len(CLUSTER_PALETTE)], label=f"Cluster {cl} (n={len(sub)})")
    axes[0].set_title(f"KMeans (k={best_k})", fontsize=11)
    axes[0].legend(loc="best", frameon=False, fontsize=8)

    for cl in range(best_k):
        sub = df[df["hier_cluster"] == cl]
        axes[1].scatter(sub["pc1"], sub["pc2"], s=10, alpha=0.6, color=CLUSTER_PALETTE[cl % len(CLUSTER_PALETTE)], label=f"Cluster {cl} (n={len(sub)})")
    axes[1].set_title(f"Hierarchical / Ward (k={best_k})", fontsize=11)
    axes[1].legend(loc="best", frameon=False, fontsize=8)

    for i, sg in enumerate(study_groups):
        sub = df[df["study_group"] == sg]
        axes[2].scatter(sub["pc1"], sub["pc2"], s=10, alpha=0.6, color=CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)], label=f"{sg} (n={len(sub)})")
    axes[2].set_title("Actual study_group", fontsize=11)
    axes[2].legend(loc="best", frameon=False, fontsize=7)

    for lv in hba1c_group_labels:
        sub = df[df["hba1c_group"] == lv]
        axes[3].scatter(sub["pc1"], sub["pc2"], s=10, alpha=0.6, color=HBA1C_GROUP_COLORS[lv], label=f"{lv} (n={len(sub)})")
    axes[3].set_title("HbA1c group (ADA thresholds)", fontsize=11)
    axes[3].legend(loc="best", frameon=False, fontsize=8)

    fig.suptitle(f"PCA projection (5 base vars + {len(CGM_FEATURES_VIF_PRUNED)} curated CGM features)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "with_cgm_pca_clusters.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)

    # --- cluster profiles ---
    profiles = []
    for method, col in [("KMeans", "kmeans_cluster"), ("Hierarchical", "hier_cluster")]:
        for cl in sorted(df[col].unique()):
            sub = df[df[col] == cl]
            row = {"method": method, "cluster": cl, "n": len(sub)}
            for c in BASE_FEATURE_COLS:
                row[VAR_LABELS[c]] = sub[c].mean()
            top_group = sub["study_group"].value_counts(normalize=True)
            row["modal_study_group"] = top_group.index[0]
            row["modal_study_group_pct"] = round(100 * top_group.iloc[0], 1)
            profiles.append(row)
    profiles_df = pd.DataFrame(profiles)
    profiles_df.to_csv(os.path.join(OUT_DIR, "with_cgm_cluster_profiles.csv"), index=False)
    print("\ncluster profiles (base variables only, for readability -- see with_cgm_pca_loadings.csv for CGM loadings):")
    print(profiles_df.to_string(index=False))

    id_col = "participant_id" if "participant_id" in df.columns else df.columns[0]
    df[[id_col, "kmeans_cluster", "hier_cluster", "study_group", "hba1c_group"]].to_csv(
        os.path.join(OUT_DIR, "with_cgm_cluster_assignments.csv"), index=False
    )
    print("\nsaved with_cgm_elbow_silhouette.png, with_cgm_dendrogram.png, with_cgm_pca_clusters.png, "
          "with_cgm_pca_loadings.png/.csv, with_cgm_cluster_profiles.csv, with_cgm_cluster_assignments.csv")


if __name__ == "__main__":
    main()
