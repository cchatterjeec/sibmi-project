"""
KMeans and hierarchical (Ward) clustering, extending clustering_by_study_group.py
by adding the 21-feature VIF-pruned/curated CGM set (regression_no_correlated_features/
select_cgm_features_vif.py) to the 5 base variables (age, HbA1c, BMI,
C-peptide, insulin) -- run separately within each study_group, same as
clustering_by_study_group.py, just with CGM features included this time.

This is a separate script from clustering_by_study_group.py -- it does not
overwrite or replace it or its outputs.

Requires the standard qc_pct_active >= 70% CGM quality filter (unlike the
base-only version, since CGM features are now part of the feature set).
Each study_group gets its own independent preprocessing (log1p on insulin/
C-peptide, z-scored within that subgroup's own distribution), its own k
selection (silhouette, swept k=2..6), and its own PCA projection (fit
separately per subgroup).

Outputs (in this directory):
  with_cgm_study_group_silhouette_selection.png
  with_cgm_study_group_pca_clusters.png
  with_cgm_study_group_cluster_profiles.csv
  with_cgm_study_group_cluster_assignments.csv
"""
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(OUT_DIR)
DATA_PATH = os.path.join(ROOT, "final_df.csv")
K_RANGE = range(2, 7)
CGM_QC_MIN_PCT_ACTIVE = 70.0

sys.path.insert(0, os.path.join(ROOT, "regression_no_correlated_features"))
from select_cgm_features_vif import CGM_FEATURES_VIF_PRUNED


def sanitize_column_name(col):
    return re.sub(r"[\[\]<]", "", col)


HBA1C_COL = sanitize_column_name("import_hba1c, Hemoglobin A1c/Hemoglobin.total in ")
CPEPTIDE_COL = sanitize_column_name("import_c_peptide, C peptide [Mass/volume] in Seru")
INSULIN_COL = sanitize_column_name("import_insulin, Insulin [Units/volume] in Serum o")
BMI_COL = "bmi_vsorres, BMI"

BASE_VAR_LABELS = {"age": "Age", HBA1C_COL: "HbA1c", BMI_COL: "BMI", CPEPTIDE_COL: "C-peptide", INSULIN_COL: "Insulin"}
BASE_FEATURE_COLS = ["age", HBA1C_COL, BMI_COL, CPEPTIDE_COL, INSULIN_COL]
FEATURE_COLS = BASE_FEATURE_COLS + CGM_FEATURES_VIF_PRUNED
LOG_COLS = {CPEPTIDE_COL, INSULIN_COL}

STUDY_GROUPS = [
    "healthy", "pre_diabetes_lifestyle_controlled",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled", "insulin_dependent",
]
GROUP_TITLES = {
    "healthy": "Healthy", "pre_diabetes_lifestyle_controlled": "Pre-diabetes",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled": "Oral medication",
    "insulin_dependent": "Insulin dependent",
}
CLUSTER_PALETTE = ["#2a78d6", "#eb6834", "#2ca858", "#a259c6", "#d6b02a", "#d64550"]

df = pd.read_csv(DATA_PATH)
df.columns = [sanitize_column_name(c) for c in df.columns]
before = len(df)
df = df[df["qc_pct_active"] >= CGM_QC_MIN_PCT_ACTIVE]
print(f"[QC filter] dropped {before - len(df)} participants with qc_pct_active < {CGM_QC_MIN_PCT_ACTIVE}% or missing")
df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
print(f"n={len(df)} (complete cases on {len(FEATURE_COLS)} features: "
      f"{len(BASE_FEATURE_COLS)} base + {len(CGM_FEATURES_VIF_PRUNED)} curated CGM)")

fig_sil, axes_sil = plt.subplots(1, 4, figsize=(20, 4.5), facecolor="#fcfcfb")
fig_pca, axes_pca = plt.subplots(2, 4, figsize=(20, 10), facecolor="#fcfcfb")

profiles, assignments = [], []

for i, group in enumerate(STUDY_GROUPS):
    sub = df[df["study_group"] == group].copy()
    X = sub[FEATURE_COLS].copy()
    for c in LOG_COLS:
        X[c] = np.log1p(X[c])
    X_scaled = StandardScaler().fit_transform(X)

    silhouettes = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_scaled)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))
    best_k = list(K_RANGE)[int(np.argmax(silhouettes))]
    print(f"\n{GROUP_TITLES[group]} (n={len(sub)}): silhouette by k = "
          f"{dict(zip(K_RANGE, [round(s, 3) for s in silhouettes]))}, chosen k={best_k}")

    ax = axes_sil[i]
    ax.set_facecolor("#fcfcfb")
    ax.plot(list(K_RANGE), silhouettes, color="#eb6834", marker="o", markersize=6, linewidth=1.6)
    ax.axvline(best_k, color="#999999", linewidth=1.0, linestyle="--", zorder=1)
    ax.set_xlabel("k"); ax.set_ylabel("Silhouette score")
    ax.set_title(f"{GROUP_TITLES[group]} (n={len(sub)})\nchosen k={best_k}", fontsize=11)
    ax.xaxis.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit(X_scaled)
    sub["kmeans_cluster"] = kmeans.labels_
    Z = linkage(X_scaled, method="ward")
    sub["hier_cluster"] = fcluster(Z, t=best_k, criterion="maxclust") - 1

    pca = PCA(n_components=2, random_state=0)
    X_pca = pca.fit_transform(X_scaled)
    sub["pc1"], sub["pc2"] = X_pca[:, 0], X_pca[:, 1]
    var_explained = pca.explained_variance_ratio_

    for row_idx, (method, col) in enumerate([("KMeans", "kmeans_cluster"), ("Hierarchical", "hier_cluster")]):
        ax = axes_pca[row_idx, i]
        ax.set_facecolor("#fcfcfb")
        for cl in sorted(sub[col].unique()):
            cl_sub = sub[sub[col] == cl]
            ax.scatter(cl_sub["pc1"], cl_sub["pc2"], s=10, alpha=0.6,
                       color=CLUSTER_PALETTE[cl % len(CLUSTER_PALETTE)], label=f"Cluster {cl} (n={len(cl_sub)})")
        title_prefix = GROUP_TITLES[group] if row_idx == 0 else ""
        ax.set_title(f"{title_prefix}\n{method} (k={best_k})" if row_idx == 0 else f"{method} (k={best_k})", fontsize=10.5)
        ax.set_xlabel(f"PC1 ({var_explained[0]:.0%})", fontsize=9)
        ax.set_ylabel(f"PC2 ({var_explained[1]:.0%})", fontsize=9)
        ax.legend(loc="best", frameon=False, fontsize=7)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    for method, col in [("KMeans", "kmeans_cluster"), ("Hierarchical", "hier_cluster")]:
        for cl in sorted(sub[col].unique()):
            cl_sub = sub[sub[col] == cl]
            row = {"study_group": GROUP_TITLES[group], "method": method, "cluster": cl, "n": len(cl_sub)}
            for c in BASE_FEATURE_COLS:
                row[BASE_VAR_LABELS[c]] = cl_sub[c].mean()
            profiles.append(row)

    id_col = "participant_id" if "participant_id" in sub.columns else sub.columns[0]
    assignments.append(sub[[id_col, "study_group", "kmeans_cluster", "hier_cluster"]])

fig_sil.suptitle(f"KMeans k selection by study_group (5 base vars + {len(CGM_FEATURES_VIF_PRUNED)} curated CGM features)", fontsize=13)
fig_sil.tight_layout()
fig_sil.savefig(os.path.join(OUT_DIR, "with_cgm_study_group_silhouette_selection.png"), dpi=200, facecolor=fig_sil.get_facecolor())
plt.close(fig_sil)

fig_pca.suptitle("Within-study_group clustering (5 base vars + curated CGM): PCA fit separately per group", fontsize=14)
fig_pca.tight_layout()
fig_pca.savefig(os.path.join(OUT_DIR, "with_cgm_study_group_pca_clusters.png"), dpi=200, facecolor=fig_pca.get_facecolor())
plt.close(fig_pca)

profiles_df = pd.DataFrame(profiles)
profiles_df.to_csv(os.path.join(OUT_DIR, "with_cgm_study_group_cluster_profiles.csv"), index=False)
print("\ncluster profiles (base variables only, for readability):")
print(profiles_df.to_string(index=False))

assignments_df = pd.concat(assignments, ignore_index=True)
assignments_df.to_csv(os.path.join(OUT_DIR, "with_cgm_study_group_cluster_assignments.csv"), index=False)

print("\nsaved with_cgm_study_group_silhouette_selection.png, with_cgm_study_group_pca_clusters.png, "
      "with_cgm_study_group_cluster_profiles.csv, with_cgm_study_group_cluster_assignments.csv")
