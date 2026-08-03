"""
Plots silhouette score vs. k for each study_group's spectral+KMeans fit
(clustering_matrix.py), reading the already-saved silhouette_by_k from each
group's meta.json. Marks the chosen best_k for each group.

Output: cluster_matrix/silhouette_by_k.png
"""
import importlib.util
import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

GROUP_LABELS = {
    "healthy": "Healthy",
    "prediabetic": "Prediabetic",
    "oral_medication": "Oral medication",
    "insulin_dependent": "Insulin dependent",
}
# Fixed categorical order/colors, validated light-mode (dataviz skill,
# validate_palette.py): ALL CHECKS PASS. The one adjacent pair in the 6-8 CVD
# floor band (green/orange) is used here with the required secondary
# encoding -- direct end-of-line labels, not color alone.
GROUP_COLORS = {
    "healthy": C.CLUSTER_PALETTE[0],
    "prediabetic": C.CLUSTER_PALETTE[1],
    "oral_medication": C.CLUSTER_PALETTE[2],
    "insulin_dependent": C.CLUSTER_PALETTE[3],
}


def main():
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for slug in C.GROUPS:
        with open(os.path.join(C.out_dir(slug), "meta.json")) as f:
            meta = json.load(f)
        sil_by_k = {int(k): v for k, v in meta["silhouette_by_k"].items()}
        ks = sorted(sil_by_k)
        scores = [sil_by_k[k] for k in ks]
        best_k = meta["best_k"]
        color = GROUP_COLORS[slug]

        ax.plot(ks, scores, color=color, linewidth=2, marker="o", markersize=8,
                markerfacecolor=color, markeredgecolor="#fcfcfb", markeredgewidth=1, zorder=2)
        # highlight the chosen k with a larger ringed marker
        ax.scatter([best_k], [sil_by_k[best_k]], s=170, facecolor=color,
                   edgecolor="#33322e", linewidth=1.5, zorder=3)
        # direct end-of-line label (secondary encoding alongside the legend,
        # since one adjacent color pair sits in the CVD floor band)
        ax.annotate(GROUP_LABELS[slug], (ks[-1], scores[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9.5, color=color)

    ax.set_xlabel("k (number of clusters)", fontsize=10.5)
    ax.set_ylabel("Silhouette score (standardized 5-var feature space)", fontsize=10.5)
    ax.set_title(
        "cluster_matrix: silhouette by k, spectral (k-NN graph) + KMeans\n"
        "ringed marker = chosen k; outliers removed before fitting",
        fontsize=12,
    )
    ax.set_xticks(list(C.K_RANGE))
    ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c9c7c0")
    ax.spines["bottom"].set_color("#c9c7c0")
    legend_handles = [
        Line2D([0], [0], color=GROUP_COLORS[s], marker="o", markersize=8,
               markerfacecolor=GROUP_COLORS[s], linewidth=2, label=GROUP_LABELS[s])
        for s in C.GROUPS
    ]
    ax.legend(
        handles=legend_handles, loc="upper right", frameon=False, fontsize=9,
        labelcolor=[GROUP_COLORS[s] for s in C.GROUPS], handlelength=1.2,
    )
    fig.tight_layout()
    out_path = os.path.join(C.OUT_ROOT, "silhouette_by_k.png")
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
