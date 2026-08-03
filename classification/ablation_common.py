"""
Shared feature-set definitions, model fitters, and bootstrap-CI helpers for
the classification (healthy vs pre-diabetic) ablation studies in this
directory.

This is a copy of /n/groups/patel/chandrima/ablation_common.py, adjusted
for classification-only use rather than importing the shared root module:
the regression-only fitters (ElasticNet/XGBoost regressors) and the R2/
stratified forest-plot functions are dropped, and a make_auroc_forest_plot
is added in their place (same vertical-forest-plot layout, AUROC on the
x-axis with a chance-level reference line at 0.5 instead of R2).

Ablation stages (cumulative):
  Model 1: Age, Sex, Race
  Model 2: Model 1 + BMI
  Model 3: Model 2 + Wearables (avg_daily_steps, avg_sleep_minutes)
  Model 4: Model 3 + Routine clinical labs (excluding EXCLUDED_GLUCOSE_FEATURES,
           serum glucose, and insulin; HbA1c is also excluded here via
           target_col since study_group -- healthy vs pre-diabetic -- is
           clinically HbA1c-based, so leaving it in would make "Model 4"
           mostly just measure whether the model can read HbA1c off the
           labs sheet rather than anything genuinely new)
  Model 5a: Model 4 + the full CGM ML feature catalog (cgm_ml_features.py /
            CGM_FEATURES.md), minus gmi_pct (an algebraic eA1c estimate --
            same near-tautology concern as HbA1c itself)
  Model 5b: Model 4 + serum glucose (alternative to Model 5a)
  Model 6a/6b: Model 3 + CGM / serum glucose, skipping routine labs entirely
"""
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cgm_ml_features import DICTIONARY as CGM_ML_DICTIONARY

DATA_PATH = "/n/groups/patel/chandrima/final_df.csv"


def sanitize_column_name(col):
    """XGBoost rejects feature names containing '[', ']', or '<'."""
    return re.sub(r"[\[\]<]", "", col)


EXCLUDED_GLUCOSE_FEATURES = [
    sanitize_column_name(c) for c in [
        "tar", "tbr",
        "import_albumin, Albumin [Mass/volume] in Serum or",
        "import_protein_total, Protein [Mass/volume] in Se",
        "import_bun, Urea nitrogen [Mass/volume] in Serum ",
        "import_total_cholesterol, Cholesterol [Mass/volum",
    ]
]
GLUCOSE_COL = sanitize_column_name("import_glucose, Glucose [Mass/volume] in Serum or")
INSULIN_COL = sanitize_column_name("import_insulin, Insulin [Units/volume] in Serum o")
CPEPTIDE_COL = sanitize_column_name("import_c_peptide, C peptide [Mass/volume] in Seru")
HBA1C_COL = sanitize_column_name("import_hba1c, Hemoglobin A1c/Hemoglobin.total in ")

# Full CGM ML feature catalog (cgm_ml_features.py), minus the QC/coverage
# columns -- those gate wear-time before modeling and aren't predictors.
CGM_ML_FEATURES = [c for c, *_ in CGM_ML_DICTIONARY if not c.startswith("qc_")]
CGM_IMPUTE_COLS = CGM_ML_FEATURES

MODEL_TYPE_STYLE = {
    "ElasticNet": {"color": "#2a78d6", "marker": "o"},
    "XGBoost": {"color": "#eb6834", "marker": "s"},
}

CGM_QC_MIN_PCT_ACTIVE = 70.0  # Battelino 2019 consensus floor for reliable CGM-derived metrics


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [sanitize_column_name(c) for c in df.columns]
    # Drop participants whose CGM wear is too sparse to trust (qc_pct_active
    # < 70%). Participants with NO CGM match at all (qc_pct_active is NaN)
    # are left alone -- unaffected by this quality gate, still get their CGM
    # columns median-imputed below.
    before = len(df)
    df = df[df["qc_pct_active"].isna() | (df["qc_pct_active"] >= CGM_QC_MIN_PCT_ACTIVE)]
    dropped = before - len(df)
    if dropped:
        print(f"[load_data] dropped {dropped} participants with qc_pct_active < {CGM_QC_MIN_PCT_ACTIVE}%")
    for col in CGM_IMPUTE_COLS:
        medians = df.groupby("split")[col].transform("median")
        df[col] = df[col].fillna(medians)
    return df


def get_ablation_stages(df, exclude_cpeptide=False, include_no_labs_variants=False, target_col=None,
                         cgm_features=None):
    """Return an ordered list of (stage_name, feature_list) tuples.

    If include_no_labs_variants, two extra stages are appended that branch
    directly off Model 3 (demographics + BMI + wearables), adding CGM
    metrics or serum glucose without routine labs in between.

    target_col, if given, is excluded from the routine-labs feature set so
    it never leaks in as its own predictor.

    cgm_features, if given, overrides CGM_ML_FEATURES (the full 67-feature
    catalog) as the CGM block used in stages 5a/6a."""
    race_cols = sorted(c for c in df.columns if c.startswith("race_"))
    sex_cols = sorted(c for c in df.columns if c.startswith("sex_"))
    demo = ["age"] + race_cols + sex_cols
    bmi = ["bmi_vsorres, BMI"]
    wearables = ["avg_daily_steps", "avg_sleep_minutes"]

    import_cols = [c for c in df.columns if c.startswith("import_")]
    excluded = set(EXCLUDED_GLUCOSE_FEATURES) | {GLUCOSE_COL, INSULIN_COL}
    if exclude_cpeptide:
        excluded.add(CPEPTIDE_COL)
    if target_col is not None:
        excluded.add(target_col)
    routine_labs = [c for c in import_cols if c not in excluded]

    cgm = cgm_features if cgm_features is not None else CGM_ML_FEATURES
    serum_glucose = [GLUCOSE_COL]

    stage1 = demo
    stage2 = stage1 + bmi
    stage3 = stage2 + wearables
    stage4 = stage3 + routine_labs
    stage5a = stage4 + cgm
    stage5b = stage4 + serum_glucose

    stages = [
        ("Model 1: Age, Sex, Race", stage1),
        ("Model 2: + BMI", stage2),
        ("Model 3: + Wearables", stage3),
        ("Model 4: + Routine Labs", stage4),
        ("Model 5a: + CGM Metrics", stage5a),
        ("Model 5b: + Serum Glucose", stage5b),
    ]
    if include_no_labs_variants:
        stages += [
            ("Model 6a: Model 3 + CGM (no labs)", stage3 + cgm),
            ("Model 6b: Model 3 + Serum Glucose (no labs)", stage3 + serum_glucose),
        ]
    return stages


def bootstrap_ci(y_true, y_score, metric_fn, n_boot=1000, seed=0):
    """Nonparametric bootstrap CI (2.5/97.5 percentile) for a metric evaluated
    on paired (y_true, y_score) arrays via resampling test-set rows."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if yt.min() == yt.max():
            continue
        try:
            stats.append(metric_fn(yt, ys))
        except (ValueError, ZeroDivisionError):
            continue
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return lo, hi


def fit_elasticnet_classifier(X_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegressionCV(
            Cs=10, cv=5, penalty="elasticnet", solver="saga",
            l1_ratios=[0.1, 0.5, 0.7, 0.9, 1.0], max_iter=5000,
            scoring="average_precision", n_jobs=-1, random_state=0,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def fit_xgb_classifier(X_train, y_train):
    base = XGBClassifier(eval_metric="aucpr", random_state=0, n_jobs=1)
    grid = {"max_depth": [2, 3, 4], "n_estimators": [100, 200], "learning_rate": [0.03, 0.1]}
    gs = GridSearchCV(base, grid, scoring="average_precision", cv=5, n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_


def fit_elasticnet_regressor(X_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 1.0], cv=5, max_iter=10000,
            n_jobs=-1, random_state=0,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def fit_xgb_regressor(X_train, y_train):
    base = XGBRegressor(random_state=0, n_jobs=1)
    grid = {"max_depth": [2, 3, 4], "n_estimators": [100, 200], "learning_rate": [0.03, 0.1]}
    gs = GridSearchCV(base, grid, scoring="r2", cv=5, n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_


def make_auroc_forest_plot(results_df, out_path, title):
    """One point + 95% CI per (stage, model_type); stages on the y-axis,
    AUROC on the x-axis, chance-level (0.5) reference line."""
    stages = results_df["stage"].unique().tolist()
    model_types = list(MODEL_TYPE_STYLE.keys())
    n_stages = len(stages)
    y_base = np.arange(n_stages)[::-1]  # first stage at top
    dodge = {model_types[0]: 0.12, model_types[1]: -0.12}

    fig, ax = plt.subplots(figsize=(8, 1.05 * n_stages + 1.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.axvline(0.5, color="#999999", linewidth=1.0, linestyle="--", zorder=1)
    ax.text(0.505, n_stages - 0.5, "chance", color="#999999", fontsize=8.5, va="center")

    for mt in model_types:
        sub = results_df[results_df["model_type"] == mt].set_index("stage").loc[stages]
        y = y_base + dodge[mt]
        x = sub["auroc"].values
        lo = x - sub["auroc_lo"].values
        hi = sub["auroc_hi"].values - x
        ax.errorbar(
            x, y, xerr=[lo, hi], fmt=MODEL_TYPE_STYLE[mt]["marker"],
            color=MODEL_TYPE_STYLE[mt]["color"], ecolor=MODEL_TYPE_STYLE[mt]["color"],
            elinewidth=1.4, capsize=3, markersize=9, markeredgecolor="#0b0b0b",
            markeredgewidth=0.6, label=mt, zorder=3,
        )

    ax.set_yticks(y_base)
    ax.set_yticklabels(stages, fontsize=10)
    ax.set_xlim(0.35, 1.0)
    ax.set_xlabel("AUROC", fontsize=11, color="#0b0b0b")
    ax.set_title(title, fontsize=13, color="#0b0b0b", pad=12)
    ax.xaxis.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
