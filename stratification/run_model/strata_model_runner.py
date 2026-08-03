"""
Runs the fixed "glucose labs in, CGM out" insulin-prediction feature set
(ablation rung 3 from ablation_runner.py) separately within each stratum of
a demographic variable x study_group -- e.g. weight_class x study_group,
race x study_group, or sex x study_group -- instead of within study_group
alone.

Unlike ablation_runner.py (which relies on a pre-existing 'split' column
shared across the whole dataset), each stratum here gets its own fresh
train_test_split: the global split was only balanced against study_group,
so it isn't guaranteed to leave a workable train/test mix once the data is
sliced this much finer.

This module holds the shared modeling code; weight_class_model.py,
race_model.py, and sex_model.py are thin wrappers that each just pick
which strata column to group on.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from ablation_runner import (
    ADMIN_COLS,
    ENET_PARAM_GRID,
    TARGET_COL,
    XGB_PARAM_GRID,
    bootstrap_r2_se,
    build_default_feature_sets,
    clean_col,
)

# Derived grouping/label columns that must never be used as model features.
NON_FEATURE_DERIVED_COLS = {
    "weight_class", "strata_race", "strata_sex", "strata_weight_class",
}

MIN_STRATUM_SIZE = 101


def build_feature_cols(df):
    """
    The fixed "glucose labs in, CGM out" feature set -- i.e. ablation rung
    "3_labs_only_remove_cgm" from ablation_runner.build_default_feature_sets
    (fasting glucose + HbA1c included; tir/cv_glucose and the always-excluded
    glucose columns dropped) -- with the weight_class/strata_* label columns
    stripped out as well, since those are grouping labels, not predictors.
    """
    feature_cols = build_default_feature_sets(df)["3_labs_only_remove_cgm"]
    return [c for c in feature_cols if c not in NON_FEATURE_DERIVED_COLS]


def _fit_one_stratum(stratum_df, feature_cols, target_col, cv, test_size, random_state):
    stratum_df = stratum_df.replace([np.inf, -np.inf], np.nan)
    stratum_df = stratum_df.dropna(subset=feature_cols + [target_col])

    cat_cols = [c for c in ("sex", "race") if c in feature_cols]
    X = pd.get_dummies(stratum_df[feature_cols], columns=cat_cols, drop_first=True)
    y = stratum_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    result = {"n": len(stratum_df), "n_train": len(X_train), "n_test": len(X_test)}

    # ----- ElasticNet (GridSearchCV over a Pipeline) -----
    enet_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("enet", ElasticNet(random_state=random_state, max_iter=10000)),
    ])
    enet_grid = GridSearchCV(
        estimator=enet_pipeline,
        param_grid=ENET_PARAM_GRID,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=8,
    )
    enet_grid.fit(X_train, y_train)
    enet_preds = enet_grid.predict(X_test)

    result["enet_best_params"] = enet_grid.best_params_
    result["enet_r2"] = r2_score(y_test, enet_preds)
    result["enet_r2_se"] = bootstrap_r2_se(y_test, enet_preds, random_state=random_state)
    result["enet_mse"] = mean_squared_error(y_test, enet_preds)

    # ----- XGBoost (GridSearchCV) -----
    X_train_xgb = X_train.copy()
    X_test_xgb = X_test.copy()
    X_train_xgb.columns = [clean_col(c) for c in X_train_xgb.columns]
    X_test_xgb.columns = [clean_col(c) for c in X_test_xgb.columns]

    xgb_grid = GridSearchCV(
        estimator=XGBRegressor(random_state=random_state, n_jobs=1),
        param_grid=XGB_PARAM_GRID,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=8,
    )
    xgb_grid.fit(X_train_xgb, y_train)
    xgb_preds = xgb_grid.predict(X_test_xgb)

    result["xgb_best_params"] = xgb_grid.best_params_
    result["xgb_r2"] = r2_score(y_test, xgb_preds)
    result["xgb_r2_se"] = bootstrap_r2_se(y_test, xgb_preds, random_state=random_state)
    result["xgb_mse"] = mean_squared_error(y_test, xgb_preds)

    return result


def run_strata_ablation(
    df,
    strata_col,
    output_dir,
    target_col=TARGET_COL,
    cv=5,
    test_size=0.2,
    random_state=42,
    min_stratum_size=MIN_STRATUM_SIZE,
):
    feature_cols = build_feature_cols(df)

    divider = "=" * 60
    print(divider)
    print(f"Stratifying by: {strata_col}")
    print(f"  {len(feature_cols)} features: {feature_cols}")
    print(divider)

    summary_rows = {}
    for stratum in df[strata_col].dropna().unique():
        stratum_df = df[df[strata_col] == stratum].copy()
        n_total = len(stratum_df)

        print(f"\n--- {strata_col} = {stratum} ---")
        if n_total < min_stratum_size:
            print(f"Skipping - only {n_total} rows (< {min_stratum_size} minimum)")
            continue

        n_available = stratum_df.dropna(subset=feature_cols + [target_col]).shape[0]
        if n_available < min_stratum_size:
            print(f"Skipping - only {n_available} rows after dropping NAs (< {min_stratum_size} minimum)")
            continue

        result = _fit_one_stratum(stratum_df, feature_cols, target_col, cv, test_size, random_state)

        print(f"  n = {result['n']}  (train={result['n_train']}, test={result['n_test']})")
        print(f"  ElasticNet  best_params={result['enet_best_params']}  "
              f"R2={result['enet_r2']:.4f} (SE={result['enet_r2_se']:.4f})  MSE={result['enet_mse']:.4f}")
        print(f"  XGBoost     best_params={result['xgb_best_params']}  "
              f"R2={result['xgb_r2']:.4f} (SE={result['xgb_r2_se']:.4f})  MSE={result['xgb_mse']:.4f}")

        summary_rows[stratum] = result

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows).T
    summary_df.index.name = strata_col
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path)

    print("\n" + "=" * 60)
    print(f"All strata complete. Summary written to {summary_path}")
    print("=" * 60)
    print(summary_df[["n", "n_train", "n_test", "enet_r2", "enet_r2_se", "xgb_r2", "xgb_r2_se"]])

    return summary_df


def main(strata_col, default_output_dir):
    parser = argparse.ArgumentParser(
        description=f"Run insulin-prediction models within each {strata_col} stratum."
    )
    parser.add_argument("--data", default="strata_df.csv", help="Path to input CSV")
    parser.add_argument("--output-dir", default=default_output_dir, help="Directory to write results to")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds for GridSearchCV")
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out test fraction within each stratum")
    parser.add_argument(
        "--min-stratum-size", type=int, default=MIN_STRATUM_SIZE,
        help="Skip a stratum with fewer than this many rows",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    run_strata_ablation(
        df,
        strata_col,
        output_dir=args.output_dir,
        cv=args.cv,
        test_size=args.test_size,
        min_stratum_size=args.min_stratum_size,
    )
