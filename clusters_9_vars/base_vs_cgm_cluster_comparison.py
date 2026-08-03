"""
Stage 1d: does adding CGM features to the clustering (clustering_with_cgm.py)
reshuffle who's grouped with whom, or mostly agree with the base clustering
(clustering_base.py, 9-variable panel) while refining it?

Output: base_vs_cgm_crosstab.csv, base_vs_cgm_crosstab.png
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import adjusted_rand_score

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

base = pd.read_csv(os.path.join(OUT_DIR, "clustering_base_assignments.csv"))
cgm = pd.read_csv(os.path.join(OUT_DIR, "clustering_with_cgm_assignments.csv"))

merged = base[["participant_id", "kmeans_cluster"]].merge(
    cgm[["participant_id", "kmeans_cluster"]], on="participant_id", suffixes=("_base", "_cgm")
)
print(f"n={len(merged)} participants present in both clusterings")

ari = adjusted_rand_score(merged["kmeans_cluster_base"], merged["kmeans_cluster_cgm"])
print(f"Adjusted Rand Index (base clusters vs with-CGM clusters) = {ari:.3f}")

crosstab = pd.crosstab(merged["kmeans_cluster_base"], merged["kmeans_cluster_cgm"])
print("\nBase cluster (rows) vs with-CGM cluster (cols) crosstab:")
print(crosstab)
crosstab.to_csv(os.path.join(OUT_DIR, "base_vs_cgm_crosstab.csv"))

target_base = merged[merged["kmeans_cluster_base"] == 2]
print(f"\nBase cluster 2 (kidney-complication phenotype, n={len(target_base)}) -> with-CGM cluster distribution:")
print(target_base["kmeans_cluster_cgm"].value_counts().sort_index())

fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
im = ax.imshow(crosstab.values, cmap="Blues")
for i in range(crosstab.shape[0]):
    for j in range(crosstab.shape[1]):
        ax.text(j, i, str(crosstab.values[i, j]), ha="center", va="center",
                  color="white" if crosstab.values[i, j] > crosstab.values.max() / 2 else "black")
ax.set_xticks(range(crosstab.shape[1])); ax.set_xticklabels([f"with-CGM {c}" for c in crosstab.columns])
ax.set_yticks(range(crosstab.shape[0])); ax.set_yticklabels([f"Base {c}" for c in crosstab.index])
ax.set_title(f"Base clusters vs with-CGM clusters (ARI={ari:.3f})", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "base_vs_cgm_crosstab.png"), dpi=200, facecolor=fig.get_facecolor())
plt.close(fig)
print("\nsaved base_vs_cgm_crosstab.csv, base_vs_cgm_crosstab.png")
