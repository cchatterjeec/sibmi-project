"""
Runs SHAP value analysis for the fixed "glucose labs in, CGM out" insulin-
prediction feature set (ablation rung 3 from ablation_runner.py) separately
within each stratum of a demographic variable x study_group -- e.g.
weight_class x study_group, race x study_group, or sex x study_group.

Each stratum gets its own XGBoost model (GridSearchCV-tuned, same param grid
as strata_model_runner.py) fit on a fresh train/test split, then a
shap.TreeExplainer computes SHAP values on the held-out test rows. For every
stratum this writes a SHAP summary (beeswarm) plot PNG plus the raw SHAP
values as a CSV.

This module holds the shared code; shap_sex_model.py, shap_race_model.py,
and shap_weight_class_model.py are thin wrappers that each just pick which
strata column to group on -- mirroring strata_model_runner.py.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor

from ablation_runner import TARGET_COL, XGB_PARAM_GRID, clean_col
from strata_model_runner import MIN_STRATUM_SIZE, build_feature_cols


def _fit_and_explain_one_stratum(stratum_df, feature_cols, target_col, cv, test_size, random_state):
    stratum_df = stratum_df.replace([np.inf, -np.inf], np.nan)
    stratum_df = stratum_df.dropna(subset=feature_cols + [target_col])

    cat_cols = [c for c in ("sex", "race") if c in feature_cols]
    X = pd.get_dummies(stratum_df[feature_cols], columns=cat_cols, drop_first=True)
    X.columns = [clean_col(c) for c in X.columns]
    y = stratum_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    xgb_grid = GridSearchCV(
        estimator=XGBRegressor(random_state=random_state, n_jobs=1),
        param_grid=XGB_PARAM_GRID,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=8,
    )
    xgb_grid.fit(X_train, y_train)
    best_model = xgb_grid.best_estimator_

    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer(X_test)

    return {
        "n": len(stratum_df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "best_params": xgb_grid.best_params_,
        "shap_values": shap_values,
        "X_test": X_test,
    }


def run_strata_shap(
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
    print(f"SHAP analysis stratified by: {strata_col}")
    print(f"  {len(feature_cols)} features: {feature_cols}")
    print(divider)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

        result = _fit_and_explain_one_stratum(
            stratum_df, feature_cols, target_col, cv, test_size, random_state
        )
        print(f"  n = {result['n']}  (train={result['n_train']}, test={result['n_test']})")
        print(f"  best_params={result['best_params']}")

        stratum_slug = str(stratum).replace("/", "_").replace(" ", "_")

        # ---- SHAP summary (beeswarm) plot ----
        plt.figure()
        shap.summary_plot(result["shap_values"], result["X_test"], show=False)
        plt.title(f"{strata_col} = {stratum}  (n_test={result['n_test']})")
        plot_path = out_dir / f"shap_summary_{stratum_slug}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Wrote {plot_path}")

        # ---- raw SHAP values ----
        shap_df = pd.DataFrame(result["shap_values"].values, columns=result["X_test"].columns)
        shap_df.insert(0, "participant_id", stratum_df.loc[result["X_test"].index, "participant_id"].values)
        shap_csv_path = out_dir / f"shap_values_{stratum_slug}.csv"
        shap_df.to_csv(shap_csv_path, index=False)

        summary_rows[stratum] = {
            "n": result["n"],
            "n_train": result["n_train"],
            "n_test": result["n_test"],
            "best_params": result["best_params"],
        }

    summary_df = pd.DataFrame(summary_rows).T
    summary_df.index.name = strata_col
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path)

    print("\n" + "=" * 60)
    print(f"All strata complete. Summary written to {summary_path}")
    print(f"SHAP plots written to {out_dir}/")
    print("=" * 60)

    return summary_df


def main(strata_col, default_output_dir):
    parser = argparse.ArgumentParser(
        description=f"Run SHAP analysis for insulin prediction within each {strata_col} stratum."
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
    run_strata_shap(
        df,
        strata_col,
        output_dir=args.output_dir,
        cv=args.cv,
        test_size=args.test_size,
        min_stratum_size=args.min_stratum_size,
    )
