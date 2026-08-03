"""
Residual diagnostics (CGM features): same fit as residuals_vs_covariates.py
(ElasticNet, Model 6a, CGM: no correlated features) -- checks whether the
prediction error (residual = actual - predicted HbA1c) correlates with
each of the 21 individual curated CGM features used in this model, the
same way residuals_vs_covariates.py checked age/BMI/steps/sleep: scatter +
LOWESS smoother + Spearman rho + p-value, one panel per feature.

With 21 tests run at once, a couple of nominal p<0.05 hits are expected by
chance alone -- the Bonferroni-corrected threshold (0.05/21 = 2.4e-3) is
the more honest bar for "this one's probably real," and is reported
alongside the raw p-value for every feature. Panel borders are colored to
flag this: green = survives Bonferroni, orange = nominal p<0.05 only,
gray = not significant.

Outputs (in this directory):
  residuals_vs_cgm_features.png   -- grid of scatter+smoother panels
  residuals_vs_cgm_features.csv   -- rho, p-value, Bonferroni flag, sorted by |rho|
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from statsmodels.nonparametric.smoothers_lowess import lowess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ablation_common import (
    HBA1C_COL,
    fit_elasticnet_regressor,
    get_ablation_stages,
    load_data,
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from select_cgm_features_vif import CGM_FEATURES_VIF_PRUNED

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

df = load_data()
stages = get_ablation_stages(
    df, exclude_cpeptide=True, include_no_labs_variants=True, target_col=HBA1C_COL,
    cgm_features=CGM_FEATURES_VIF_PRUNED,
)
stage_name, feats = [s for s in stages if s[0].startswith("Model 6a")][0]

train = df[df["split"] == "train"]
test = df[df["split"] == "test"]
X_train, X_test = train[feats], test[feats]
y_train, y_test = train[HBA1C_COL], test[HBA1C_COL]

model = fit_elasticnet_regressor(X_train, y_train)
pred = model.predict(X_test)
residual = y_test.values - pred
print(f"ElasticNet, {stage_name}: R2={r2_score(y_test, pred):.3f}")

n_tests = len(CGM_FEATURES_VIF_PRUNED)
bonferroni = 0.05 / n_tests
print(f"n CGM features tested: {n_tests}, Bonferroni threshold = {bonferroni:.2e}")

rows = []
for col in CGM_FEATURES_VIF_PRUNED:
    x = test[col].values
    rho, pval = spearmanr(x, residual)
    rows.append({"cgm_feature": col, "rho": rho, "p": pval, "bonferroni_significant": pval < bonferroni})
results_df = pd.DataFrame(rows).sort_values("rho", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
results_df.to_csv(os.path.join(OUT_DIR, "residuals_vs_cgm_features.csv"), index=False)
print(results_df.to_string(index=False))
print(f"...{(results_df['p'] < 0.05).sum()} of {n_tests} nominal p<0.05, "
      f"{results_df['bonferroni_significant'].sum()} survive Bonferroni")

ncols = 4
nrows = int(np.ceil(n_tests / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows), facecolor="#fcfcfb")
axes = axes.flatten()

for ax, col in zip(axes, CGM_FEATURES_VIF_PRUNED):
    x = test[col].values
    rho, pval = spearmanr(x, residual)
    if pval < bonferroni:
        border_color, border_width = "#2ca858", 2.2
    elif pval < 0.05:
        border_color, border_width = "#eb6834", 1.6
    else:
        border_color, border_width = "#cccccc", 0.8

    ax.set_facecolor("#fcfcfb")
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--", zorder=1)
    ax.scatter(x, residual, s=8, alpha=0.4, color="#2a78d6", edgecolor="none", zorder=2)
    try:
        smoothed = lowess(residual, x, frac=0.6, return_sorted=True)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="#eb6834", linewidth=1.8, zorder=3)
    except Exception:
        pass

    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(border_width)
    ax.set_title(f"{col}\n$\\rho$={rho:+.2f}, p={pval:.1e}", fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])

for ax in axes[n_tests:]:
    ax.axis("off")

fig.suptitle(
    f"Residual vs. CGM feature diagnostics -- ElasticNet, {stage_name}\n"
    f"CGM: no correlated features  |  green border = Bonferroni-significant (p<{bonferroni:.1e}), "
    f"orange = nominal p<0.05 only",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT_DIR, "residuals_vs_cgm_features.png"), dpi=180, facecolor=fig.get_facecolor())
print("saved residuals_vs_cgm_features.png")
