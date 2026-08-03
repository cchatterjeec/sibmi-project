"""
Residual diagnostics: for the same fit as actual_vs_predicted_hba1c.py
(ElasticNet, Model 6a, CGM: no correlated features), check whether the
prediction error (residual = actual - predicted HbA1c) correlates with
continuous covariates already used elsewhere in the ablation stages (age,
BMI, avg_daily_steps, avg_sleep_minutes) -- i.e. does the model
systematically over/under-predict for older/younger, heavier/leaner,
more/less active, or more/less sleep participants, rather than being
unbiased with respect to these.

Each panel: covariate (x) vs. residual (y), a LOWESS smoother, a y=0
reference line (no bias), and Spearman rho + p-value.

Output: residuals_vs_covariates.png
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
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
COVARIATES = {
    "age": "Age (years)",
    "bmi_vsorres, BMI": "BMI",
    "avg_daily_steps": "Avg. daily steps",
    "avg_sleep_minutes": "Avg. sleep (minutes)",
}

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
print(f"residual: mean={residual.mean():.3f}, sd={residual.std():.3f}")

fig, axes = plt.subplots(2, 2, figsize=(11, 9), facecolor="#fcfcfb")
axes = axes.flatten()

for ax, (col, label) in zip(axes, COVARIATES.items()):
    x = test[col].values
    rho, pval = spearmanr(x, residual)
    print(f"{label:<20s} Spearman rho={rho:+.3f}  p={pval:.3e}")

    ax.set_facecolor("#fcfcfb")
    ax.axhline(0, color="#999999", linewidth=1.0, linestyle="--", zorder=1)
    ax.scatter(x, residual, s=14, alpha=0.5, color="#2a78d6", edgecolor="none", zorder=2)

    smoothed = lowess(residual, x, frac=0.5, return_sorted=True)
    ax.plot(smoothed[:, 0], smoothed[:, 1], color="#eb6834", linewidth=2.2, zorder=3)

    ax.set_xlabel(label, fontsize=10.5)
    ax.set_ylabel("Residual (actual − predicted HbA1c)", fontsize=9.5)
    ax.set_title(f"Spearman $\\rho$ = {rho:+.3f}  (p = {pval:.2e})", fontsize=11)
    ax.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

fig.suptitle(
    f"Residual vs. covariate diagnostics -- ElasticNet, {stage_name}\nCGM: no correlated features",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "residuals_vs_covariates.png"), dpi=200, facecolor=fig.get_facecolor())
print("saved residuals_vs_covariates.png")
