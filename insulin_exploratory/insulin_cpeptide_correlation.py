"""
Scatter plots of serum insulin vs. c-peptide, one panel per study_group, with
Spearman rank correlation (rho) and its p-value annotated on each panel.
Spearman (not Pearson) is used since it doesn't assume a linear relationship
or normally-distributed variables -- just monotonic association. Uses every
participant (train+test combined) since nothing is being fit.
"""
import os
import sys

import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ablation_common import CPEPTIDE_COL, INSULIN_COL, load_data

STUDY_GROUPS = [
    "healthy",
    "pre_diabetes_lifestyle_controlled",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled",
    "insulin_dependent",
]
GROUP_LABELS = {
    "healthy": "Healthy",
    "pre_diabetes_lifestyle_controlled": "Pre-diabetic",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled": "Oral/non-insulin med",
    "insulin_dependent": "Insulin-dependent",
}

df = load_data()
df = df.dropna(subset=[INSULIN_COL, CPEPTIDE_COL])

fig, axes = plt.subplots(2, 2, figsize=(10, 9), facecolor="#fcfcfb")
for ax, group in zip(axes.flat, STUDY_GROUPS):
    sub = df[df["study_group"] == group]
    rho, p = spearmanr(sub[INSULIN_COL], sub[CPEPTIDE_COL])

    ax.set_facecolor("#fcfcfb")
    ax.scatter(sub[INSULIN_COL], sub[CPEPTIDE_COL], s=12, alpha=0.4, color="#2a78d6", edgecolors="none")
    ax.set_xlabel("Serum insulin")
    ax.set_ylabel("C-peptide")
    ax.set_title(f"{GROUP_LABELS[group]} (n={len(sub)})", fontsize=11)
    ax.text(
        0.05, 0.95, f"Spearman $\\rho$={rho:.3f}\np={p:.2e}",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="#ccc", alpha=0.8),
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    print(f"{group:75s} rho={rho:.3f}  p={p:.2e}  n={len(sub)}")

fig.suptitle("Insulin vs. c-peptide correlation by study group", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "insulin_cpeptide_correlation.png"),
    dpi=200, facecolor=fig.get_facecolor(),
)
