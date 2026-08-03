#!/usr/bin/env python3
"""
select_cgm_features_vif.py -- CGM feature list used by every regression
script in this directory (the "no correlated features" variant), using
ONLY the CGM feature block (demographics, wearables, routine labs, and
serum glucose are untouched everywhere else in this directory's
regressions).

CGM_FEATURES_VIF_PRUNED now comes from
cgm_feature_selection.select_cgm_features_curated() (repo root):
domain-priority greedy pairwise selection (walk the 67 features in a fixed
priority order -- ATTD-consensus TIR family first, then the doc's own
recommended family representatives, then the complexity/spectral/circadian
families, then episode/dynamics, then distribution shape; keep a feature
only if |r| <= 0.8 with everything already kept), followed by a residual
VIF check (threshold 10) to catch multi-way collinearity a pairwise check
alone can miss. See that module's docstring for the full rationale and the
domain-priority tier list.

This module used to freeze the result of a much blunter method
(select_low_vif_cgm_features, still below for reference/comparison): plain
iterative "drop the single highest-VIF feature" with no domain-priority
input. That approach mechanically dropped tir_70_180_pct, mean_glucose, and
cv_pct entirely, purely because of removal order -- not defensible for a
clinically-interpretable SHAP output. Superseded here; kept as a function
so the two methods stay comparable, but CGM_FEATURES_VIF_PRUNED (the name
imported by every other script in this directory) is the curated result.
"""
import os
import sys

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ablation_common import CGM_ML_FEATURES, load_data
from cgm_feature_selection import select_cgm_features_curated

VIF_THRESHOLD = 10.0


def compute_vif(X):
    Xc = add_constant(X, has_constant="add")
    vifs = [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])]
    return pd.Series(vifs, index=X.columns)


def select_low_vif_cgm_features(verbose=True):
    """Superseded plain-greedy-VIF method -- see module docstring. Kept only
    for reference/comparison against the curated selection actually used."""
    df = load_data()
    train = df[df["split"] == "train"]
    X = train[CGM_ML_FEATURES].copy()

    zero_var = [c for c in X.columns if X[c].std() == 0]
    if zero_var:
        if verbose:
            print(f"[drop] zero-variance on train: {zero_var}")
        X = X.drop(columns=zero_var)

    X = (X - X.mean()) / X.std()

    dropped = []
    while True:
        vif = compute_vif(X)
        worst = vif.idxmax()
        if vif[worst] <= VIF_THRESHOLD or X.shape[1] <= 1:
            break
        if verbose:
            print(f"[drop] {worst:28s} VIF={vif[worst]:10.1f}  ({X.shape[1]} -> {X.shape[1]-1} remaining)")
        dropped.append((worst, vif[worst]))
        X = X.drop(columns=[worst])

    kept = sorted(X.columns.tolist())
    if verbose:
        final_vif = compute_vif(X)
        print(f"\n[kept] {len(kept)} of {len(CGM_ML_FEATURES)} CGM features (VIF <= {VIF_THRESHOLD}):")
        for c in kept:
            print(f"   {c:28s} VIF={final_vif[c]:6.2f}")
        print(f"\n[dropped] {len(dropped) + len(zero_var)} features:")
        for c in zero_var:
            print(f"   {c}  (zero-variance on train)")
        for c, v in dropped:
            print(f"   {c}  (VIF={v:.1f} at removal)")
    return kept


# Result of select_cgm_features_curated() (domain-priority greedy pairwise
# selection + residual VIF check), frozen here so the four scripts in this
# directory share one fixed list instead of each re-running the selection
# on import. Regenerate by running cgm_feature_selection.py directly if
# cgm_ml_features.py or final_df.csv change.
#
# Regenerated 2026-07-22 after fixing the cosinor_acrophase_h acrophase
# bug (see cgm_ml_features.py's cosinor() docstring) and re-merging the
# corrected CGM features into final_df.csv. "modd" dropped out of this run
# (r=0.801 with tir_70_180_pct, just over the 0.8 threshold -- previously
# just under it); every other feature selected the same as before.
CGM_FEATURES_VIF_PRUNED = [
    "between_day_sd",
    "cosinor_acrophase_h",
    "cosinor_amplitude",
    "dawn_rise",
    "dfa_alpha",
    "kurtosis",
    "mag_per_hour",
    "mean_hyper_dur_min",
    "mean_hypo_dur_min",
    "min_glucose",
    "n_hyper_events",
    "nocturnal_hypo_pct",
    "sample_entropy",
    "skewness",
    "spectral_circadian_frac",
    "spectral_dominant_period_h",
    "spectral_entropy",
    "tbr_lt54_pct",
    "tbr_lt70_pct",
    "tir_70_180_pct",
]


if __name__ == "__main__":
    select_low_vif_cgm_features(verbose=True)
    print()
    select_cgm_features_curated(verbose=True)
