"""
Stage 1b: pairwise Euclidean distance matrix on the standardized 5-base-
variable space (clustering_base.py's preprocessing), rows/columns sorted
by KMeans cluster label, visualized as a heatmap.

Clean dark (low-distance) blocks along the diagonal per cluster, with
lighter (high-distance) off-diagonal regions, would mean the clusters are
genuinely separating people. Fuzzy blocks that bleed into neighboring
clusters mean weak separation -- consistent with (and a visual version of)
the silhouette score already computed in clustering_base.py.

Output: distance_heatmap.png
"""
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = "/n/groups/patel/chandrima/final_df.csv"


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

assignments = pd.read_csv(os.path.join(OUT_DIR, "clustering_base_assignments.csv"))
id_col = "participant_id" if "participant_id" in df.columns else df.columns[0]
df = df.merge(assignments[[id_col, "kmeans_cluster"]], on=id_col)
print(f"n={len(df)}")

X = df[FEATURE_COLS].copy()
for c in LOG_COLS:
    X[c] = np.log1p(X[c])
X_scaled = StandardScaler().fit_transform(X)

order = df.sort_values("kmeans_cluster").index.to_numpy()
dist_matrix = squareform(pdist(X_scaled, metric="euclidean"))
dist_sorted = dist_matrix[np.ix_(order, order)]
cluster_sorted = df.loc[order, "kmeans_cluster"].to_numpy()

fig, ax = plt.subplots(figsize=(8.5, 7.5), facecolor="#fcfcfb")
im = ax.imshow(dist_sorted, cmap="viridis", aspect="auto")
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Euclidean distance (standardized 5-var space)", fontsize=10)

# Draw cluster boundary lines
boundaries = np.where(np.diff(cluster_sorted) != 0)[0] + 1
for b in boundaries:
    ax.axhline(b, color="white", linewidth=1.2)
    ax.axvline(b, color="white", linewidth=1.2)

tick_positions, tick_labels = [], []
start = 0
for cl in sorted(df["kmeans_cluster"].unique()):
    count = (cluster_sorted == cl).sum()
    tick_positions.append(start + count / 2)
    tick_labels.append(f"Cluster {cl}\n(n={count})")
    start += count
ax.set_xticks(tick_positions); ax.set_xticklabels(tick_labels, fontsize=9)
ax.set_yticks(tick_positions); ax.set_yticklabels(tick_labels, fontsize=9)
ax.set_title("Pairwise distance matrix, sorted by KMeans cluster\n(darker = closer together)", fontsize=12.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "distance_heatmap.png"), dpi=200, facecolor=fig.get_facecolor())
plt.close(fig)
print("saved distance_heatmap.png")
