"""
Residual diagnostics (categorical covariates): same fit as
residuals_vs_covariates.py (ElasticNet, Model 6a, CGM: no correlated
features) -- checks whether the prediction error (residual = actual -
predicted HbA1c) differs by sex, race, BMI weight class, or study_group
(diabetes/medication status), using the Mann-Whitney U test (a.k.a.
Wilcoxon rank-sum test) since residual distributions aren't assumed
normal.

Sex: direct two-group rank-sum test (Female vs Male).
Race / weight class / study_group: one-vs-rest rank-sum test for each
category with test-set n >= 15 (a handful of race categories and
"Underweight" in weight class have far too few test-set participants --
0-7, or 1 for Underweight -- for a stable rank-sum test, so they're
excluded here rather than reported with unstable p-values).

weight_class comes from stratification/strata_df.csv (same file used by
eval_model_6a_stratified.py in this directory: Underweight <18.5, Normal
18.5-25, Overweight 25-30, Obesity >=30 kg/m^2), joined by participant_id.

Output: residuals_vs_categorical_covariates.png (2x2: Sex, Race,
Weight class, Study group)
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
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
MIN_GROUP_N = 15
PALETTE = ["#2a78d6", "#eb6834", "#2ca858", "#a259c6", "#d6b02a"]

LABEL_OVERRIDES = {
    "Black or African American": "Black/African\nAmerican",
    "Other race, ethnicity or origin": "Other race/\nethnicity",
    "White or Caucasian": "White/\nCaucasian",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled": "Oral\nmedication",
    "pre_diabetes_lifestyle_controlled": "Pre-diabetes",
    "insulin_dependent": "Insulin\ndependent",
    "healthy": "Healthy",
}
STUDY_GROUP_ORDER = [
    "healthy", "pre_diabetes_lifestyle_controlled",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled", "insulin_dependent",
]
BMI_CATEGORY_ORDER = ["Underweight", "Normal", "Overweight", "Obesity"]

df = load_data()
weight_class = pd.read_csv(
    os.path.join(OUT_DIR, "..", "stratification", "strata_df.csv"),
    usecols=["participant_id", "weight_class"],
)
df = df.merge(weight_class, on="participant_id", how="left")

stages = get_ablation_stages(
    df, exclude_cpeptide=True, include_no_labs_variants=True, target_col=HBA1C_COL,
    cgm_features=CGM_FEATURES_VIF_PRUNED,
)
stage_name, feats = [s for s in stages if s[0].startswith("Model 6a")][0]

train = df[df["split"] == "train"]
test = df[df["split"] == "test"].copy()
X_train, X_test = train[feats], test[feats]
y_train, y_test = train[HBA1C_COL], test[HBA1C_COL]

model = fit_elasticnet_regressor(X_train, y_train)
pred = model.predict(X_test)
test["residual"] = y_test.values - pred
print(f"ElasticNet, {stage_name}: R2={r2_score(y_test, pred):.3f}")


def label_for(level):
    return LABEL_OVERRIDES.get(level, str(level))


def draw_one_vs_rest_panel(ax, col, order, title):
    levels = [lv for lv in order if lv in test[col].dropna().unique()] if order else \
        sorted(test[col].dropna().unique().tolist())
    groups, labels, pvals = [], [], []
    for lv in levels:
        mask = test[col] == lv
        n = int(mask.sum())
        if n < MIN_GROUP_N:
            print(f"Skipping {col}={lv}: n={n} < {MIN_GROUP_N}")
            continue
        in_group, out_group = test.loc[mask, "residual"].values, test.loc[~mask, "residual"].values
        _, p = mannwhitneyu(in_group, out_group, alternative="two-sided")
        print(f"{title}: {lv} (n={n}) vs rest (n={len(test) - n})  rank-sum p={p:.3e}")
        groups.append(in_group)
        labels.append(f"{label_for(lv)}\n(n={n})")
        pvals.append(p)

    ax.set_facecolor("#fcfcfb")
    bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True, showfliers=False, widths=0.5)
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color); patch.set_alpha(0.35)
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        jitter = rng.uniform(-0.08, 0.08, size=len(g))
        ax.scatter(np.full(len(g), i + 1) + jitter, g, s=10, alpha=0.4, color="#333333", zorder=3)
    ymax = max(np.concatenate(groups))
    for i, p in enumerate(pvals):
        ax.text(i + 1, ymax + 0.3, f"p={p:.3f}", ha="center", fontsize=9)
    ax.axhline(0, color="#999999", linewidth=1.0, linestyle="--", zorder=1)
    ax.set_title(f"{title} (one-vs-rest rank-sum test)", fontsize=11.5)
    ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.setp(ax.get_xticklabels(), fontsize=8.5)


fig, axes = plt.subplots(2, 2, figsize=(13, 11), facecolor="#fcfcfb")

# --- Sex: direct two-group comparison ---
ax = axes[0, 0]
ax.set_facecolor("#fcfcfb")
female_res = test.loc[test["sex_Female"] == 1, "residual"].values
male_res = test.loc[test["sex_Male"] == 1, "residual"].values
_, p_sex = mannwhitneyu(female_res, male_res, alternative="two-sided")
print(f"Sex: Female (n={len(female_res)}) vs Male (n={len(male_res)})  rank-sum p={p_sex:.3e}")

groups = [female_res, male_res]
labels = [f"Female\n(n={len(female_res)})", f"Male\n(n={len(male_res)})"]
bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True, showfliers=False, widths=0.5)
for patch, color in zip(bp["boxes"], PALETTE):
    patch.set_facecolor(color); patch.set_alpha(0.35)
rng = np.random.default_rng(0)
for i, g in enumerate(groups):
    jitter = rng.uniform(-0.08, 0.08, size=len(g))
    ax.scatter(np.full(len(g), i + 1) + jitter, g, s=10, alpha=0.4, color="#333333", zorder=3)
ax.axhline(0, color="#999999", linewidth=1.0, linestyle="--", zorder=1)
ax.set_ylabel("Residual (actual − predicted HbA1c)", fontsize=10.5)
ax.set_title(f"Sex\nrank-sum p = {p_sex:.3f}", fontsize=11.5)
ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# --- Race, Weight class, Study group: one-vs-rest ---
race_cols_present = sorted(c for c in df.columns if c.startswith("race_"))
test["race"] = test[race_cols_present].idxmax(axis=1).str.replace("race_", "", regex=False)
draw_one_vs_rest_panel(axes[0, 1], "race", None, "Race")
draw_one_vs_rest_panel(axes[1, 0], "weight_class", BMI_CATEGORY_ORDER, "Weight class")
draw_one_vs_rest_panel(axes[1, 1], "study_group", STUDY_GROUP_ORDER, "Study group")
axes[1, 0].set_ylabel("Residual (actual − predicted HbA1c)", fontsize=10.5)

fig.suptitle(
    f"Residual vs. categorical covariates -- ElasticNet, {stage_name}\nCGM: no correlated features",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "residuals_vs_categorical_covariates.png"), dpi=200, facecolor=fig.get_facecolor())
print("saved residuals_vs_categorical_covariates.png")
