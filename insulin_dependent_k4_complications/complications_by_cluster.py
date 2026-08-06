"""
Exploratory: are the insulin_dependent forced-k=4 clusters (spectral
affinity + KMeans on age/BMI/insulin/C-peptide/HbA1c -- see
cluster_matrix/clusters_3d_k4_insulin_dependent.py) associated with any of
the complications-adjacent variables actually present in this dataset?

There is no diagnosed-retinopathy/nephropathy/neuropathy field, no CKD
stage, no amputation or cardiovascular-event field anywhere in the raw
data (checked measurement.csv's full 108 variables + the demographics
export + data dictionary). The six variables here are the complete set of
complications-adjacent measurements available:
  - uacr (urine albumin/creatinine ratio) -- kidney/nephropathy proxy
  - serum creatinine, BUN/creatinine ratio -- kidney function
  - NT-proBNP, Troponin T -- cardiac strain/injury proxies
  - foot_worst (Semmes-Weinstein monofilament, worse of the 2 feet,
    0-10 correctly-felt sites) -- peripheral neuropathy screening

This is exploratory and NOT a validated clinical analysis: n per cluster
is small (32-76), no multiple-comparison correction is applied (6 tests,
so treat p<0.05 as suggestive, not confirmatory -- see PRINTED note), and
the "reduced sensation" foot cutoff (<8/10) is a practical approximation,
not a cutoff validated for this specific instrument/scoring convention.

Outputs (in this directory):
  complications_summary.csv           -- per-cluster n/median/IQR + Kruskal-Wallis
                                          (or chi-square for foot_reduced) per marker
  complications_violin_by_cluster.png -- one panel per continuous marker
  neuropathy_reduced_pct_by_cluster.png -- % with foot_worst < 8, by cluster
"""
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal

from common import (
    BUN_CREATININE_COL,
    CLUSTER_PALETTE,
    CREATININE_COL,
    LOG_COLS,
    MARKER_COMPLICATION,
    MARKER_LABELS,
    NT_PROBNP_COL,
    TROPONIN_COL,
    load_merged,
)

# Distinguishes the 3 complication categories at a glance in the violin
# grid, independent of the (unrelated) per-cluster CLUSTER_PALETTE colors.
COMPLICATION_COLORS = {
    "Kidney (nephropathy)": "#7a4f9e",
    "Cardiac": "#c0392b",
    "Neuropathy": "#1f7a5c",
}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_CLUSTERS = 4

# Brief characterization from cluster_profiles_k4.csv, for plot titles only.
CLUSTER_BLURB = {
    0: "older, lower BMI/insulin",
    1: "younger, best glycemic control",
    2: "high HbA1c (poor control)",
    3: "highest BMI/insulin/C-peptide",
}

CONTINUOUS_MARKERS = ["uacr", CREATININE_COL, BUN_CREATININE_COL, NT_PROBNP_COL, TROPONIN_COL, "foot_worst"]
FOOT_REDUCED_THRESHOLD = 8  # correctly-felt sites out of 10; <8 flagged as reduced sensation


def main():
    df = load_merged()
    print(f"n={len(df)} across {N_CLUSTERS} clusters: "
          f"{dict(df['cluster'].value_counts().sort_index())}")

    # --- per-marker Kruskal-Wallis across clusters + summary table ---
    summary_rows = []
    for marker in CONTINUOUS_MARKERS:
        groups = [df.loc[df["cluster"] == cl, marker].dropna().values for cl in range(N_CLUSTERS)]
        stat, pval = kruskal(*groups)
        print(f"{MARKER_LABELS[marker]:42s} Kruskal-Wallis H={stat:.2f}  p={pval:.4f}"
              f"{'  <-- p<0.05' if pval < 0.05 else ''}")
        for cl, g in enumerate(groups):
            summary_rows.append({
                "marker": MARKER_LABELS[marker], "cluster": cl, "cluster_blurb": CLUSTER_BLURB[cl],
                "n": len(g), "median": np.median(g) if len(g) else np.nan,
                "q25": np.percentile(g, 25) if len(g) else np.nan,
                "q75": np.percentile(g, 75) if len(g) else np.nan,
                "kruskal_H": stat, "kruskal_p": pval,
            })

    # --- neuropathy as a binary "reduced sensation" flag + chi-square ---
    df["foot_reduced"] = df["foot_worst"] < FOOT_REDUCED_THRESHOLD
    ct = pd.crosstab(df["cluster"], df["foot_reduced"])
    chi2, chi2_p, dof, expected = chi2_contingency(ct)
    print(f"\n{'Foot reduced sensation (<'+str(FOOT_REDUCED_THRESHOLD)+'/10)':42s} "
          f"chi2={chi2:.2f}  dof={dof}  p={chi2_p:.4f}{'  <-- p<0.05' if chi2_p < 0.05 else ''}")
    pct_reduced = df.groupby("cluster")["foot_reduced"].mean() * 100
    n_valid = df.groupby("cluster")["foot_reduced"].apply(lambda s: s.notna().sum())
    for cl in range(N_CLUSTERS):
        summary_rows.append({
            "marker": "Foot reduced sensation (% with worse-foot score <8/10)", "cluster": cl,
            "cluster_blurb": CLUSTER_BLURB[cl], "n": int(n_valid[cl]),
            "median": pct_reduced[cl], "q25": np.nan, "q75": np.nan,
            "kruskal_H": chi2, "kruskal_p": chi2_p,
        })
        print(f"  cluster {cl} ({CLUSTER_BLURB[cl]}): {pct_reduced[cl]:.1f}% reduced (n={int(n_valid[cl])})")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, "complications_summary.csv"), index=False)
    print(f"\nNote: 6 markers tested with no multiple-comparison correction -- "
          f"treat any single p<0.05 as suggestive, not confirmatory.")

    # --- violin grid, one panel per continuous marker ---
    n_markers = len(CONTINUOUS_MARKERS)
    fig, axes = plt.subplots(1, n_markers, figsize=(3.6 * n_markers, 4.9), facecolor="#fcfcfb")
    for marker, ax in zip(CONTINUOUS_MARKERS, axes):
        ax.set_facecolor("#fcfcfb")
        vals_by_cluster = [df.loc[df["cluster"] == cl, marker].dropna().values for cl in range(N_CLUSTERS)]
        plot_vals = [np.log1p(v) if marker in LOG_COLS else v for v in vals_by_cluster]
        parts = ax.violinplot(plot_vals, showmeans=True, showextrema=False)
        for cl, body in enumerate(parts["bodies"]):
            color = CLUSTER_PALETTE[cl]
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.65)
        parts["cmeans"].set_color("#33322e")
        ax.set_xticks(range(1, N_CLUSTERS + 1))
        ax.set_xticklabels([f"C{cl}\n(n={len(vals_by_cluster[cl])})" for cl in range(N_CLUSTERS)], fontsize=8.5)
        stat_row = summary_df[summary_df["marker"] == MARKER_LABELS[marker]]
        pval = stat_row["kruskal_p"].iloc[0]
        label = MARKER_LABELS[marker] + (" (log1p)" if marker in LOG_COLS else "")
        category = MARKER_COMPLICATION[marker]
        cat_color = COMPLICATION_COLORS[category]
        # Category is folded into the title itself (one Text object) rather
        # than a separate ax.annotate above it -- tight_layout only reserves
        # space for the title/axes system, so anything placed via annotate
        # at an axes-fraction y>1 gets silently overlapped instead of
        # properly spaced.
        ax.set_title(f"{category.upper()}\n{label}\nKruskal-Wallis p={pval:.3f}", fontsize=9.5, pad=8)
        ax.title.set_color(cat_color)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        "insulin_dependent, k=4 clusters: complications-adjacent markers\n"
        "(exploratory -- 6 markers, no multiple-comparison correction)", fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(os.path.join(OUT_DIR, "complications_violin_by_cluster.png"), dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # --- neuropathy bar chart ---
    fig, ax = plt.subplots(figsize=(8.5, 5.8), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    bars = ax.bar(range(N_CLUSTERS), [pct_reduced[cl] for cl in range(N_CLUSTERS)],
                   color=CLUSTER_PALETTE[:N_CLUSTERS], alpha=0.85, edgecolor="#0b0b0b", linewidth=0.6,
                   width=0.6)
    for cl, bar in zip(range(N_CLUSTERS), bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{pct_reduced[cl]:.0f}%\n(n={int(n_valid[cl])})", ha="center", fontsize=9.5)
    ax.set_xticks(range(N_CLUSTERS))
    wrapped_labels = [
        f"Cluster {cl}\n" + "\n".join(textwrap.wrap(CLUSTER_BLURB[cl], width=16))
        for cl in range(N_CLUSTERS)
    ]
    ax.set_xticklabels(wrapped_labels, fontsize=8.5)
    ax.set_xlim(-0.7, N_CLUSTERS - 0.3)
    ax.set_ylabel(f"% with worse-foot monofilament score < {FOOT_REDUCED_THRESHOLD}/10", fontsize=10.5)
    ax.set_ylim(0, max(pct_reduced.max() * 1.3, 10))
    ax.set_title(
        f"insulin_dependent, k=4: reduced foot sensation by cluster\n"
        f"chi-square p={chi2_p:.3f} (exploratory, cutoff is a practical approximation)",
        fontsize=11.5, pad=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "neuropathy_reduced_pct_by_cluster.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"\nsaved complications_summary.csv, complications_violin_by_cluster.png, "
          f"neuropathy_reduced_pct_by_cluster.png")


if __name__ == "__main__":
    main()
