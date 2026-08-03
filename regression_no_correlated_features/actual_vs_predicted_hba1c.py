"""
Actual vs. predicted HbA1c scatter plot for Model 6a (+ CGM Metrics, no
routine labs -- isolates the CGM effect on its own), ElasticNet -- R2=0.654,
per hba1c_regression_results.csv, VIF-pruned/curated 21-feature CGM set,
the better of the two model types at this stage (XGBoost: R2=0.639) --
refit here on the same full population and train/test split as
run_hba1c_regression_ablation.py.

Points are colored by study_group (diabetes/medication status). R2/
Spearman/best-fit line are always computed on the full test set (all
groups combined), regardless of axis range.

Outputs (in this directory):
  actual_vs_predicted_hba1c.png         -- full data range
  actual_vs_predicted_hba1c_zoomed.png  -- same, axes clipped to 4-10% (where
                                            most of the cohort actually sits)
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, spearmanr
from sklearn.metrics import r2_score

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

STUDY_GROUP_COLORS = {
    "healthy": "#2ca02c",
    "pre_diabetes_lifestyle_controlled": "#d6b02a",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled": "#1f77b4",
    "insulin_dependent": "#d62728",
}
STUDY_GROUP_LABELS = {
    "healthy": "Healthy",
    "pre_diabetes_lifestyle_controlled": "Pre-diabetes",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled": "Oral medication",
    "insulin_dependent": "Insulin dependent",
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
study_group = test["study_group"].values

r2 = r2_score(y_test, pred)
slope, intercept, _, _, _ = linregress(y_test, pred)
rho, pval = spearmanr(y_test, pred)
print(f"ElasticNet, {stage_name}: R2={r2:.3f}")
print(f"OLS fit: predicted = {slope:.3f} * actual + {intercept:.3f}")
print(f"Spearman rho={rho:.3f}, p={pval:.3e}")


def make_figure(out_name, fixed_lims=None):
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    lims = fixed_lims if fixed_lims is not None else [min(y_test.min(), pred.min()) - 0.3, max(y_test.max(), pred.max()) + 0.3]

    ax.plot(lims, lims, color="#999999", linewidth=1.0, linestyle="--", zorder=1, label="Perfect prediction")
    fit_x = np.array(lims)
    ax.plot(fit_x, slope * fit_x + intercept, color="#333333", linewidth=2.0, zorder=2, label=f"Best fit (slope={slope:.2f})")

    for group, color in STUDY_GROUP_COLORS.items():
        mask = study_group == group
        if mask.sum() == 0:
            continue
        ax.scatter(y_test[mask], pred[mask], s=16, alpha=0.55, color=color, edgecolor="none",
                   zorder=3, label=f"{STUDY_GROUP_LABELS[group]} (n={mask.sum()})")

    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("Actual HbA1c (%)", fontsize=11, color="#0b0b0b")
    ax.set_ylabel("Predicted HbA1c (%)", fontsize=11, color="#0b0b0b")
    ax.set_title(
        f"Actual vs. predicted HbA1c -- ElasticNet, Model 6a\n(CGM: no correlated features)\n"
        f"$R^2$ = {r2:.3f}   |   Spearman $\\rho$ = {rho:.3f} (p = {pval:.2e})",
        fontsize=11.5, color="#0b0b0b", pad=12,
    )
    ax.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, out_name), dpi=200, facecolor=fig.get_facecolor())
    print(f"saved {out_name}")


make_figure("actual_vs_predicted_hba1c.png")
make_figure("actual_vs_predicted_hba1c_zoomed.png", fixed_lims=[4, 10])
