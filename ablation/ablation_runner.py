"""
Ablation study runner for insulin prediction.

Fits ElasticNet and XGBoost (each tuned with GridSearchCV) separately within
each study_group, for one or more feature-set "scenarios". Meant to isolate
where glucose-derived features (fasting glucose, HbA1c, and CGM metrics like
time-in-range / coefficient of variation) add predictive value for insulin.

Usage (import as a function):

    from ablation_runner import run_ablation, build_default_feature_sets
    import pandas as pd

    df = pd.read_csv("final_df.csv")
    feature_sets = build_default_feature_sets(df)
    run_ablation(feature_sets, df=df, output_dir="ablation_results")

Usage (command line, e.g. on O2 via sbatch run_ablation.sh):

    python3 ablation_runner.py --data final_df.csv --output-dir ablation_results

You can also pass your own scenarios instead of the defaults -- run_ablation
just needs a dict of {scenario_name: [list of column names]}.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

TARGET_COL = "import_insulin, Insulin [Units/volume] in Serum o"

GLUCOSE_LAB_FEATURES = [
    "import_glucose, Glucose [Mass/volume] in Serum or",
    "import_hba1c, Hemoglobin A1c/Hemoglobin.total in ",
]
CGM_FEATURES = ["tir", "cv_glucose"]
# Glucose-related columns that are excluded from every scenario (never added
# back in), as opposed to CGM_FEATURES which get added back in scenario 3.
EXCLUDED_GLUCOSE_FEATURES = ["tar", "tbr",
    "import_albumin, Albumin [Mass/volume] in Serum or",
    "import_protein_total, Protein [Mass/volume] in Se",
    "import_bun, Urea nitrogen [Mass/volume] in Serum ",
    "import_total_cholesterol, Cholesterol [Mass/volum" ]

ADMIN_COLS = {"participant_id", "study_group", "split", TARGET_COL}

# Keys are prefixed for the "enet" step of the StandardScaler+ElasticNet
# Pipeline (see _fit_one_group) so scaling is refit inside each CV fold
# instead of leaking test-fold statistics through a pre-fit scaler.
ENET_PARAM_GRID = {
    "enet__alpha": [0.001, 0.01, 0.1, 1, 10, 100],
    "enet__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
}

XGB_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.7, 1.0],
}


def clean_col(c):
    """xgboost rejects '[', ']', '<', '>' in feature names."""
    return re.sub(r"[\[\]<>]", "", c)


def bootstrap_r2_se(y_true, y_pred, n_boot=2000, random_state=42):
    """
    Standard error of R2 on a single held-out test set, via bootstrap
    resampling of the (y_true, y_pred) pairs. A point-estimate R2 from one
    train/test split has no uncertainty attached on its own; this resamples
    the test set with replacement n_boot times and takes the std of the
    resulting R2 distribution as its SE.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(random_state)

    boot_r2 = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        boot_r2[i] = r2_score(yt, yp) if np.var(yt) > 0 else np.nan

    return np.nanstd(boot_r2, ddof=1)


def build_default_feature_sets(df):
    """
    Builds the 3-rung glucose ablation ladder for insulin prediction:

      1. no_glucose      -- drop every glucose-related var (fasting glucose,
                             HbA1c, tir, cv_glucose). tar/tbr are excluded from
                             every scenario entirely (never added back in).
      2. labs_only        -- add back fasting glucose + HbA1c, CGM still removed
      3. labs_plus_cgm    -- add back CGM metrics on top of (2): the "add CGM
                             variables" step -- time-in-range (tir) and
                             coefficient of variation (cv_glucose) only.

    "base_features" (demographics, vitals, non-glucose labs, activity/sleep)
    are the same across all three; only the glucose-related block changes.
    """
    glucose_related = set(GLUCOSE_LAB_FEATURES) | set(CGM_FEATURES) | set(EXCLUDED_GLUCOSE_FEATURES)

    base_features = [
        c
        for c in df.columns
        if c not in ADMIN_COLS
        and c not in glucose_related
        and not c.startswith("race_")
        and not c.startswith("sex_")
    ]

    return {
        "1_no_glucose": base_features,
        "2_cgm_only_remove_labs": base_features + CGM_FEATURES,
        "3_labs_only_remove_cgm": base_features + GLUCOSE_LAB_FEATURES,
        "4_labs_plus_cgm": base_features + GLUCOSE_LAB_FEATURES + CGM_FEATURES,
        
    }


def _fit_one_group(group_df, feature_cols, target_col, cv, random_state):
    group_df = group_df.replace([np.inf, -np.inf], np.nan)
    group_df = group_df.dropna(subset=feature_cols + [target_col])

    result = {"n": len(group_df)}

    cat_cols = [c for c in ("sex", "race") if c in feature_cols]
    X = pd.get_dummies(group_df[feature_cols], columns=cat_cols, drop_first=True)
    y = group_df[target_col]

    train_mask = group_df["split"] == "train"
    test_mask = group_df["split"] == "test"

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # ----- ElasticNet (GridSearchCV over a Pipeline) -----
    # StandardScaler lives inside the pipeline (rather than being fit once on
    # all of X_train beforehand) so each inner CV fold refits the scaler on
    # only its own training portion -- otherwise the held-out fold's
    # statistics leak into the scaling applied to it.
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
    enet_best = enet_grid.best_estimator_

    result["enet_best_params"] = enet_grid.best_params_
    result["enet_r2"] = r2_score(y_test, enet_preds)
    result["enet_r2_se"] = bootstrap_r2_se(y_test, enet_preds, random_state=random_state)
    result["enet_mse"] = mean_squared_error(y_test, enet_preds)
    enet_coefs = pd.Series(enet_best.named_steps["enet"].coef_, index=X.columns)

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
    xgb_best = xgb_grid.best_estimator_

    result["xgb_best_params"] = xgb_grid.best_params_
    result["xgb_r2"] = r2_score(y_test, xgb_preds)
    result["xgb_r2_se"] = bootstrap_r2_se(y_test, xgb_preds, random_state=random_state)
    result["xgb_mse"] = mean_squared_error(y_test, xgb_preds)
    xgb_importances = pd.Series(xgb_best.feature_importances_, index=X_train_xgb.columns)

    test_ids = group_df.loc[test_mask, "participant_id"].values
    predictions = pd.DataFrame(
        {
            "participant_id": test_ids,
            "actual_insulin": y_test.values,
            "pred_insulin_enet": enet_preds,
            "pred_insulin_xgb": xgb_preds,
        }
    )

    return result, enet_coefs, xgb_importances, predictions


def run_one_scenario(
    df,
    scenario_name,
    feature_cols,
    output_dir,
    target_col=TARGET_COL,
    cv=5,
    random_state=42,
    min_group_size=20,
):
    # Safety net: strip participant_id/study_group/split/target even if a
    # custom feature_sets dict accidentally includes them.
    feature_cols = [c for c in feature_cols if c not in ADMIN_COLS and c != target_col]

    divider = "=" * 60
    print(divider)
    print(f"Scenario: {scenario_name}")
    print(f"  {len(feature_cols)} features: {feature_cols}")
    print(divider)

    summary_rows = {}
    enet_coef_cols = {}
    xgb_importance_cols = {}
    prediction_rows = []

    for group in df["study_group"].dropna().unique():
        group_df = df[df["study_group"] == group].copy()
        n_available = group_df.dropna(subset=feature_cols + [target_col]).shape[0]

        print(f"\n--- Study group: {group} ---")
        if n_available < min_group_size:
            print(f"Skipping - only {n_available} rows after dropping NAs (too few to model)")
            continue

        result, enet_coefs, xgb_importances, predictions = _fit_one_group(
            group_df, feature_cols, target_col, cv, random_state
        )
        predictions.insert(1, "study_group", group)
        prediction_rows.append(predictions)

        print(f"  n = {result['n']}")
        print(f"  ElasticNet  best_params={result['enet_best_params']}  "
              f"R2={result['enet_r2']:.4f} (SE={result['enet_r2_se']:.4f})  MSE={result['enet_mse']:.4f}")
        print(f"  XGBoost     best_params={result['xgb_best_params']}  "
              f"R2={result['xgb_r2']:.4f} (SE={result['xgb_r2_se']:.4f})  MSE={result['xgb_mse']:.4f}")

        summary_rows[group] = result
        enet_coef_cols[group] = enet_coefs
        xgb_importance_cols[group] = xgb_importances

    scenario_dir = Path(output_dir) / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows).T
    summary_df.to_csv(scenario_dir / "summary.csv")

    if prediction_rows:
        pd.concat(prediction_rows, ignore_index=True).to_csv(
            scenario_dir / "predictions.csv", index=False
        )

    pd.DataFrame(enet_coef_cols).to_csv(scenario_dir / "enet_coefficients.csv")
    pd.DataFrame(xgb_importance_cols).to_csv(scenario_dir / "xgb_feature_importances.csv")

    print(f"\nWrote scenario outputs to {scenario_dir}/")
    return summary_df


def run_ablation(
    feature_sets,
    df=None,
    data_path="final_df.csv",
    output_dir="ablation_results",
    target_col=TARGET_COL,
    cv=5,
    random_state=42,
    min_group_size=20,
):
    """
    feature_sets: dict of {scenario_name: [feature_column_names]}
    df: optional pre-loaded DataFrame (otherwise read from data_path)
    """
    if df is None:
        df = pd.read_csv(data_path)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    for scenario_name, feature_cols in feature_sets.items():
        summary_df = run_one_scenario(
            df,
            scenario_name,
            feature_cols,
            output_dir,
            target_col=target_col,
            cv=cv,
            random_state=random_state,
            min_group_size=min_group_size,
        )
        summary_df = summary_df.reset_index().rename(columns={"index": "study_group"})
        summary_df.insert(0, "scenario", scenario_name)
        comparison_rows.append(summary_df)

    comparison_df = pd.concat(comparison_rows, ignore_index=True)
    comparison_path = Path(output_dir) / "ablation_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print("\n" + "=" * 60)
    print("All scenarios complete. Comparison table:")
    print("=" * 60)
    print(comparison_df[[
        "scenario", "study_group", "n",
        "enet_r2", "enet_r2_se", "enet_mse",
        "xgb_r2", "xgb_r2_se", "xgb_mse",
    ]])
    print(f"\nFull comparison written to {comparison_path}")

    return comparison_df


def main():
    parser = argparse.ArgumentParser(description="Run insulin-prediction glucose-feature ablation studies.")
    parser.add_argument("--data", default="final_df.csv", help="Path to input CSV")
    parser.add_argument("--output-dir", default="ablation_results", help="Directory to write results to")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds for GridSearchCV")
    parser.add_argument("--min-group-size", type=int, default=20, help="Skip a study_group if fewer rows remain after dropna")
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Subset of default scenario names to run (default: all). "
             "Choices: 1_no_glucose, 2_labs_only_remove_cgm, 3_labs_plus_cgm",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    feature_sets = build_default_feature_sets(df)

    if args.scenarios:
        feature_sets = {k: v for k, v in feature_sets.items() if k in args.scenarios}

    run_ablation(
        feature_sets,
        df=df,
        output_dir=args.output_dir,
        cv=args.cv,
        min_group_size=args.min_group_size,
    )


if __name__ == "__main__":
    main()
