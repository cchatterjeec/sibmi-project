"""
Boxplots and pairwise Mann-Whitney U tests of insulin and glucose across the
4 study_group categories in final_df.csv.

Outputs:
  insulin_exploratory/insulin_boxplot.png
  insulin_exploratory/insulin_pairwise_pvalues.csv
  glucose_exploratory/glucose_boxplot.png
  glucose_exploratory/glucose_pairwise_pvalues.csv
"""
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# categorical palette slots 1-4 (blue, aqua, yellow, green)
COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300"]

GROUP_ORDER = [
    "healthy",
    "pre_diabetes_lifestyle_controlled",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled",
    "insulin_dependent",
]
GROUP_LABELS = {
    "healthy": "Healthy",
    "pre_diabetes_lifestyle_controlled": "Pre-diabetic",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled": "Oral/Non-insulin\nmed controlled",
    "insulin_dependent": "Insulin-dependent",
}

VARIABLES = {
    "insulin": {
        "column": "import_insulin, Insulin [Units/volume] in Serum o",
        "ylabel": "Insulin (Units/volume)",
        "outdir": "insulin_exploratory",
    },
    "glucose": {
        "column": "import_glucose, Glucose [Mass/volume] in Serum or",
        "ylabel": "Glucose (mg/dL)",
        "outdir": "glucose_exploratory",
    },
}


def format_pvalue(p):
    return "p<0.0001" if p < 0.0001 else f"p={p:.4f}"


def add_significance_brackets(ax, pvals, positions, y_top):
    """Draw one bracket per pairwise comparison, stacked from nearest-neighbor
    pairs (lowest) to the widest pair (highest), annotated with the
    Bonferroni-corrected p-value."""
    rng = ax.get_ylim()
    step = (rng[1] - rng[0]) * 0.09
    tick = step * 0.15
    pairs = sorted(
        pvals.itertuples(index=False),
        key=lambda row: abs(positions[row.group_2] - positions[row.group_1]),
    )
    y = y_top
    for row in pairs:
        y += step
        x1, x2 = positions[row.group_1], positions[row.group_2]
        ax.plot([x1, x1, x2, x2], [y - tick, y, y, y - tick], color="#0b0b0b", linewidth=1)
        ax.text(
            (x1 + x2) / 2, y + tick, format_pvalue(row.p_value_bonferroni),
            ha="center", va="bottom", fontsize=7.5, color="#0b0b0b",
        )
    ax.set_ylim(rng[0], y + step * 0.6)


def make_boxplot(df, column, ylabel, out_path, pvals):
    data = [df.loc[df["study_group"] == g, column].dropna() for g in GROUP_ORDER]
    labels = [f"{GROUP_LABELS[g]}\n(n={len(d)})" for g, d in zip(GROUP_ORDER, data)]
    positions = {GROUP_LABELS[g]: i + 1 for i, g in enumerate(GROUP_ORDER)}

    fig, ax = plt.subplots(figsize=(7.5, 6.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops=dict(color="#0b0b0b", linewidth=2),
        boxprops=dict(linewidth=1.2, edgecolor="#0b0b0b"),
        whiskerprops=dict(linewidth=1.2, color="#52514e"),
        capprops=dict(linewidth=1.2, color="#52514e"),
    )
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    rng = np.random.default_rng(0)
    y_max = max(d.max() for d in data)
    for i, (d, color) in enumerate(zip(data, COLORS), start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(d))
        ax.scatter(
            np.full(len(d), i) + jitter, d, s=10, color=color, alpha=0.45,
            edgecolors="none", zorder=3,
        )

    ax.set_ylabel(ylabel, color="#0b0b0b", fontsize=11)
    ax.set_title(f"{ylabel.split(' (')[0]} by study group", color="#0b0b0b", fontsize=13, pad=12)
    ax.tick_params(axis="x", labelsize=9.5, colors="#0b0b0b")
    ax.tick_params(axis="y", labelsize=9.5, colors="#52514e")
    ax.yaxis.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")

    add_significance_brackets(ax, pvals, positions, y_max)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def pairwise_pvalues(df, column):
    rows = []
    pairs = list(itertools.combinations(GROUP_ORDER, 2))
    n_comparisons = len(pairs)
    for g1, g2 in pairs:
        x = df.loc[df["study_group"] == g1, column].dropna()
        y = df.loc[df["study_group"] == g2, column].dropna()
        stat, p = mannwhitneyu(x, y, alternative="two-sided")
        rows.append({
            "group_1": GROUP_LABELS[g1],
            "group_2": GROUP_LABELS[g2],
            "n_1": len(x),
            "n_2": len(y),
            "median_1": x.median(),
            "median_2": y.median(),
            "u_statistic": stat,
            "p_value": p,
            "p_value_bonferroni": min(p * n_comparisons, 1.0),
        })
    return pd.DataFrame(rows)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base_dir, "final_df.csv"))

    for name, cfg in VARIABLES.items():
        outdir = os.path.join(base_dir, cfg["outdir"])
        os.makedirs(outdir, exist_ok=True)

        pvals = pairwise_pvalues(df, cfg["column"])
        pvals.to_csv(os.path.join(outdir, f"{name}_pairwise_pvalues.csv"), index=False)

        make_boxplot(df, cfg["column"], cfg["ylabel"], os.path.join(outdir, f"{name}_boxplot.png"), pvals)
        print(f"\n=== {name} pairwise Mann-Whitney U (Bonferroni-corrected across {len(pvals)} tests) ===")
        print(pvals.to_string(index=False))


if __name__ == "__main__":
    main()
