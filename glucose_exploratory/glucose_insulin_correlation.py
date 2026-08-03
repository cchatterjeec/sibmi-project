"""
Scatter plot of serum glucose vs. serum insulin (healthy and pre-diabetic
participants only), with Spearman rank correlation (rho) and its p-value
annotated on the plot. Spearman (not Pearson) is used since it doesn't assume
a linear relationship or normally-distributed variables -- just monotonic
association.
"""
import os
import sys

import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ablation_common import GLUCOSE_COL, INSULIN_COL, load_data

POS_LABEL, NEG_LABEL = "pre_diabetes_lifestyle_controlled", "healthy"

df = load_data()
df = df[df["study_group"].isin([NEG_LABEL, POS_LABEL])]
rho, p = spearmanr(df[GLUCOSE_COL], df[INSULIN_COL])

plt.scatter(df[GLUCOSE_COL], df[INSULIN_COL], s=12, alpha=0.4, color="#2a78d6", edgecolors="none")
smoothed = lowess(df[INSULIN_COL], df[GLUCOSE_COL], frac=0.4)
plt.plot(smoothed[:, 0], smoothed[:, 1], color="#eb6834", linewidth=2, label="LOWESS")
plt.legend(loc="upper right", frameon=False)
plt.xlabel("Serum glucose")
plt.ylabel("Serum insulin")
plt.title("Glucose vs. insulin (healthy vs. pre-diabetic)")
plt.gca().text(
    0.05, 0.95, f"Spearman $\\rho$={rho:.3f}, p={p:.2e}",
    transform=plt.gca().transAxes, va="top", fontsize=10,
    bbox=dict(facecolor="white", edgecolor="#ccc", alpha=0.8),
)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "glucose_insulin_correlation.png"), dpi=200)
print(f"Spearman rho={rho:.3f}, p={p:.2e}")
