"""
Stage 1a: bootstrap stability of the 9-variable KMeans clustering
(clustering_base.py, k=3), same method as clusters_analysis/bootstrap_stability.py:
resample with replacement 100x, refit KMeans (k=3) each time, compute
per-original-cluster co-clustering stability. Matters most for the small
(n=54) kidney-complication cluster.

Output: bootstrap_stability.png
"""
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = "/n/groups/patel/chandrima/final_df.csv"
N_BOOT = 100
BEST_K = 3


def sanitize_column_name(col):
    return re.sub(r"[\[\]<]", "", col)


CPEPTIDE_COL = sanitize_column_name("import_c_peptide, C peptide [Mass/volume] in Seru")
INSULIN_COL = sanitize_column_name("import_insulin, Insulin [Units/volume] in Serum o")
BMI_COL = "bmi_vsorres, BMI"
CREATININE_COL = sanitize_column_name("import_creatinine, Creatinine [Mass/volume] in Se")
URINE_ALBUMIN_COL = sanitize_column_name("import_urine_albumin, Albumin [Mass/volume] in Ur")
URINE_CREATININE_COL = sanitize_column_name("import_urine_creatinine, Creatinine [Mass/volume]")
TRIGLYCERIDES_COL = sanitize_column_name("import_triglycerides, Triglyceride [Mass/volume] ")
HDL_COL = sanitize_column_name("import_hdl_cholesterol, Cholesterol in HDL [Mass/")
CRP_COL = sanitize_column_name("import_crp_hs, C reactive protein [Mass/volume] i")
FEATURE_COLS = ["age", BMI_COL, INSULIN_COL, CPEPTIDE_COL, CREATININE_COL, "uacr", TRIGLYCERIDES_COL, HDL_COL, CRP_COL]
LOG_COLS = {INSULIN_COL, CPEPTIDE_COL, CREATININE_COL, "uacr", TRIGLYCERIDES_COL, CRP_COL}

df = pd.read_csv(DATA_PATH)
df.columns = [sanitize_column_name(c) for c in df.columns]
df["uacr"] = df[URINE_ALBUMIN_COL] / df[URINE_CREATININE_COL].replace(0, np.nan)
df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
n = len(df)
print(f"n={n}")

X = df[FEATURE_COLS].copy()
for c in LOG_COLS:
    X[c] = np.log1p(X[c])
X_scaled = StandardScaler().fit_transform(X)

original_labels = KMeans(n_clusters=BEST_K, n_init=10, random_state=0).fit(X_scaled).labels_
cluster_ids = sorted(set(original_labels))
print(f"original cluster sizes: {[(c, int((original_labels == c).sum())) for c in cluster_ids]}")

together_count = np.zeros((n, n), dtype=np.int32)
seen_count = np.zeros((n, n), dtype=np.int32)

rng = np.random.default_rng(0)
for b in range(N_BOOT):
    idx = rng.integers(0, n, n)
    X_boot = X_scaled[idx]
    labels_boot = KMeans(n_clusters=BEST_K, n_init=10, random_state=b).fit(X_boot).labels_

    unique_idx, first_pos = np.unique(idx, return_index=True)
    labels_for_unique = labels_boot[first_pos]

    seen_count[np.ix_(unique_idx, unique_idx)] += 1
    same = labels_for_unique[:, None] == labels_for_unique[None, :]
    together_count[np.ix_(unique_idx, unique_idx)] += same

    if (b + 1) % 20 == 0:
        print(f"  ... {b + 1}/{N_BOOT} bootstrap resamples done")

with np.errstate(invalid="ignore", divide="ignore"):
    stability_matrix = np.where(seen_count > 0, together_count / np.maximum(seen_count, 1), np.nan)

stability_per_cluster = {}
for c in cluster_ids:
    members = np.where(original_labels == c)[0]
    pairs = stability_matrix[np.ix_(members, members)]
    iu = np.triu_indices(len(members), k=1)
    stability_per_cluster[c] = np.nanmean(pairs[iu])
    print(f"Cluster {c} (n={len(members)}): mean pairwise co-clustering stability = {stability_per_cluster[c]:.3f}")

fig, ax = plt.subplots(figsize=(7, 5.5), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
sizes = [int((original_labels == c).sum()) for c in cluster_ids]
bars = ax.bar([f"Cluster {c}\n(n={s})" for c, s in zip(cluster_ids, sizes)],
              [stability_per_cluster[c] for c in cluster_ids],
              color=["#2a78d6", "#eb6834", "#2ca858"][:len(cluster_ids)])
ax.axhline(1.0, color="#999999", linewidth=1.0, linestyle="--", zorder=1)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Mean pairwise co-clustering stability\n(fraction of 100 bootstrap resamples)", fontsize=10.5)
ax.set_title(f"Bootstrap stability of 9-variable KMeans clusters (k={BEST_K}, {N_BOOT} resamples)", fontsize=12)
for bar, c in zip(bars, cluster_ids):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{stability_per_cluster[c]:.3f}", ha="center", fontsize=10)
ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "bootstrap_stability.png"), dpi=200, facecolor=fig.get_facecolor())
plt.close(fig)
print("saved bootstrap_stability.png")
