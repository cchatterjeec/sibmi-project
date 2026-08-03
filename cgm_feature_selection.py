#!/usr/bin/env python3
"""
cgm_feature_selection.py -- prune the CGM ML feature catalog (cgm_ml_features.py)
down to a low-multicollinearity subset via domain-priority greedy selection,
so downstream SHAP explanations aren't diluted by near-duplicate features
splitting credit for the same signal (e.g. mean_glucose vs gmi_pct, r=0.99999;
SD vs CV vs J-index vs M-value vs GRADE vs MAGE vs CONGA, all driven by
within-day SD -- see CGM_FEATURES.md "Redundancy / feature-selection
guidance").

Method (train split only -- feature selection must not see test data):
  Walk the 67 CGM features in a fixed domain-priority order (guideline/
  consensus metrics first, then literature-recommended family
  representatives, then the complexity/spectral/circadian families the
  catalog doc calls "largely independent... highest-value additions", then
  episode/dynamics, then distribution shape, then everything else). Keep a
  feature only if its |Pearson r| with EVERY already-kept feature is <=
  CORR_THRESHOLD; otherwise it's redundant with something higher-priority
  already in the kept set, so drop it.

  This is greedy pairwise selection, not "one representative per connected
  component" (a naive graph-clustering pass was tried first and rejected --
  connected components over a correlation graph transitively chain almost
  the entire amplitude/central-tendency family into one 44-member cluster,
  so collapsing each cluster to a single survivor threw away
  tir_70_180_pct, mean_glucose, AND cv_pct together even though the doc's
  own guidance says to keep several of those, e.g. "keep ~2, cv_pct +
  mage"; it also picked tar_gt180_pct over the more standard
  tir_70_180_pct via an alphabetical tie-break). Greedy pairwise selection
  lets multiple lower-pairwise-correlation members of a broad family
  coexist, as intended.

  A residual VIF check runs on the survivors afterward (multi-way
  collinearity a pairwise check alone can miss); any feature still above
  RESIDUAL_VIF_THRESHOLD is dropped, lowest-priority first.

Output: prints the kept/dropped lists with reasons -- paste the kept list
into ablation_common.py's CGM_ML_FEATURES_SELECTED, or use
select_cgm_features_curated() directly.
"""
import sys

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

sys.path.insert(0, "/n/groups/patel/chandrima")
from ablation_common import CGM_ML_FEATURES, load_data

CORR_THRESHOLD = 0.8
RESIDUAL_VIF_THRESHOLD = 10.0

# Explicit full order, most-preferred-to-keep first. Ties within a family
# are broken by hand (e.g. tir_70_180_pct before tar_gt180_pct -- TIR is
# the headline ATTD-consensus metric) rather than alphabetically.
PRIORITY_ORDER = [
    # Tier 0: ATTD/Battelino-2019 consensus TIR family -- the metrics
    # clinicians and reviewers expect to see, never sacrifice these.
    "tir_70_180_pct", "tbr_lt70_pct", "tbr_lt54_pct",
    "tar_gt180_pct", "tar_gt250_pct", "tight_70_140_pct",
    # Tier 1: the classic two-number summary (level + spread).
    "mean_glucose", "cv_pct",
    # Tier 2: CGM_FEATURES.md's own recommended amplitude representative.
    "mage",
    # Tier 3: doc-flagged as "more orthogonal to the SD family, higher
    # marginal value" -- between-day / longest-lag structure.
    "modd", "between_day_sd", "mag_per_hour", "conga6",
    # Tier 4: core two risk-transform indices.
    "lbgi", "hbgi",
    # Tier 5: signal complexity / spectral / circadian -- doc calls these
    # "largely independent of amplitude metrics... highest-value additions,
    # keep even if individually noisy."
    "sample_entropy", "dfa_alpha", "autocorr_lag1",
    "spectral_circadian_frac", "spectral_dominant_period_h", "spectral_entropy",
    "cosinor_amplitude", "cosinor_acrophase_h",
    # Tier 6: episode/dynamics burden -- distinct clinical concept (event
    # counts & durations, rate of change) not captured by level or spread.
    "n_hypo_events", "n_hyper_events", "mean_hypo_dur_min", "mean_hyper_dur_min",
    "roc_sd", "pct_rapid_rise", "pct_rapid_fall",
    "hyper_auc_per_day", "hypo_auc_per_day",
    "dawn_rise", "nocturnal_hypo_pct", "overnight_mean",
    # Tier 7: distribution shape / floor -- kept as robust
    # spread/asymmetry/tail descriptors distinct from mean+CV.
    "iqr_glucose", "skewness", "kurtosis", "min_glucose",
]
# Tier 99: everything else (derived/secondary composites -- gmi_pct is an
# algebraic function of mean_glucose, j_index/m_value/grade/sd_glucose
# duplicate the SD family, gri* duplicate the TAR/TBR bins, etc.), appended
# in their catalog order so the walk is still fully deterministic.
PRIORITY_ORDER += [c for c in CGM_ML_FEATURES if c not in PRIORITY_ORDER]


def compute_vif(X):
    Xc = add_constant(X, has_constant="add")
    vifs = [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])]
    return pd.Series(vifs, index=X.columns)


def select_cgm_features_curated(verbose=True):
    df = load_data()
    train = df[df["split"] == "train"]
    X = train[CGM_ML_FEATURES].copy()

    zero_var = [c for c in X.columns if X[c].std() == 0]
    if zero_var:
        if verbose:
            print(f"[drop] zero-variance on train: {zero_var}")
        X = X.drop(columns=zero_var)

    corr = X.corr().abs()

    kept, dropped = [], []
    for c in PRIORITY_ORDER:
        if c not in corr.columns:
            continue
        if not kept:
            kept.append(c)
            continue
        worst = corr.loc[c, kept].idxmax()
        worst_r = corr.loc[c, kept].max()
        if worst_r > CORR_THRESHOLD:
            dropped.append((c, worst, worst_r))
            if verbose:
                print(f"[drop] {c:28s} r={worst_r:.3f} with already-kept {worst!r}")
        else:
            kept.append(c)

    if verbose:
        print(f"\n[after greedy pairwise selection] kept {len(kept)}, dropped {len(dropped)} "
              f"(+ {len(zero_var)} zero-variance)")

    # Residual multi-way collinearity check (pairwise <=0.8 doesn't rule out
    # 3+ features jointly explaining one another).
    Xk = X[kept]
    Xk_std = (Xk - Xk.mean()) / Xk.std()
    while True:
        vif = compute_vif(Xk_std)
        worst = vif.idxmax()
        if vif[worst] <= RESIDUAL_VIF_THRESHOLD or Xk_std.shape[1] <= 1:
            break
        # drop the lowest-priority (highest PRIORITY_ORDER index) feature
        # among any still tied for worst VIF, not just whichever comes first
        tied = vif[vif == vif[worst]].index.tolist()
        worst = max(tied, key=lambda c: PRIORITY_ORDER.index(c))
        if verbose:
            print(f"[residual-VIF drop] {worst:28s} VIF={vif[worst]:.1f}")
        kept.remove(worst)
        dropped.append((worst, "residual multi-way collinearity", vif[worst]))
        Xk_std = Xk_std.drop(columns=[worst])

    final_vif = compute_vif(Xk_std)
    if verbose:
        print(f"\n[FINAL] {len(kept)} of {len(CGM_ML_FEATURES)} CGM features kept:")
        for c in sorted(kept, key=lambda c: PRIORITY_ORDER.index(c)):
            print(f"   {c:28s} priority_rank={PRIORITY_ORDER.index(c):3d}  VIF={final_vif[c]:6.2f}")
        print(f"\n[FINAL] {len(dropped) + len(zero_var)} dropped:")
        for c in zero_var:
            print(f"   {c:28s} (zero-variance on train)")
        for c, reason, val in dropped:
            print(f"   {c:28s} (redundant with {reason!r}, r/VIF={val:.2f})" if isinstance(reason, str) and reason != "residual multi-way collinearity"
                  else f"   {c:28s} ({reason}, VIF={val:.1f})")

        print("\nCGM_ML_FEATURES_SELECTED = [")
        for c in sorted(kept):
            print(f'    "{c}",')
        print("]")

    return sorted(kept)


if __name__ == "__main__":
    select_cgm_features_curated(verbose=True)
