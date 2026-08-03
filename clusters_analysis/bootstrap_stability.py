"""
Stage 1a: bootstrap stability of the base-5-variable KMeans clustering
(clustering_base.py, k=3). Resample the cohort with replacement 100x,
refit KMeans (k=3) on each resample, and for every pair of people who were
in the same cluster in the ORIGINAL fit, compute the fraction of
resamples (among those where both are present) where they're still
clustered together -- a co-clustering / consensus stability score,
averaged per original cluster.

A cluster whose members keep getting split apart across resamples is
likely an artifact of one particular random draw, not a real, robust
structure -- this matters most for the small (n=102) beta-cell-exhaustion
cluster, which is the most at-risk of being unstable given its size.

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


HBA1C_COL = sanitize_column_name("import_hba1c, Hemoglobin A1c/Hemoglobin.total in ")
CPEPTIDE_COL = sanitize_column_name("import_c_peptide, C peptide [Mass/volume] in Seru")
INSULIN_COL = sanitize_column_name("import_insulin, Insulin [Units/volume] in Serum o")
BMI_COL = "bmi_vsorres, BMI"
FEATURE_COLS = ["age", HBA1C_COL, BMI_COL, CPEPTIDE_COL, INSULIN_COL]
LOG_COLS = {CPEPTIDE_COL, INSULIN_COL}

df = pd.read_csv(DATA_PATH)
df.columns = [sanitize_column_name(c) for c in df.columns]
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

# together_count[i, j] = number of resamples where i and j both appear AND end up in the
# same cluster; seen_count[i, j] = number of resamples where both i and j appear at all.
together_count = np.zeros((n, n), dtype=np.int32)
seen_count = np.zeros((n, n), dtype=np.int32)

rng = np.random.default_rng(0)
for b in range(N_BOOT):
    idx = rng.integers(0, n, n)
    X_boot = X_scaled[idx]
    labels_boot = KMeans(n_clusters=BEST_K, n_init=10, random_state=b).fit(X_boot).labels_

    # Map boot cluster labels back onto original point identities (idx may repeat --
    # use the first occurrence's label per unique original index drawn this round).
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
ax.set_title(f"Bootstrap stability of base-5-variable KMeans clusters (k={BEST_K}, {N_BOOT} resamples)", fontsize=12)
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
