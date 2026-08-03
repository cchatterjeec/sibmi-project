"""
Regression ablation (healthy + pre-diabetic only): predict serum insulin using
ElasticNet and XGBoost across 6 cumulative feature-ablation stages, restricted
to study_group in {healthy, pre_diabetes_lifestyle_controlled}. c-peptide is
excluded from the routine-labs feature set (co-secreted with insulin -> leakage).

Outputs (in this directory):
  regression_results_healthy_prediabetic.csv -- point R2 estimates + 95% bootstrap CIs
  insulin_r2_forest_healthy_prediabetic.png   -- forest plot, R2 on x-axis, ablation stage on y-axis
"""
import os
import sys

import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ablation_common import (
    INSULIN_COL,
    bootstrap_ci,
    fit_elasticnet_regressor,
    fit_xgb_regressor,
    get_ablation_stages,
    load_data,
)
from run_regression_ablation import make_r2_forest_plot

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_GROUPS = ["healthy", "pre_diabetes_lifestyle_controlled"]


def main():
    df = load_data()
    df = df[df["study_group"].isin(STUDY_GROUPS)].copy()
    stages = get_ablation_stages(df, exclude_cpeptide=True)

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    y_train_full, y_test_full = train[INSULIN_COL], test[INSULIN_COL]

    fitters = {"ElasticNet": fit_elasticnet_regressor, "XGBoost": fit_xgb_regressor}

    rows = []
    for stage_name, feats in stages:
        X_train, X_test = train[feats], test[feats]
        for model_type, fitter in fitters.items():
            model = fitter(X_train, y_train_full)
            pred_test = model.predict(X_test)
            r2 = r2_score(y_test_full.values, pred_test)
            r2_lo, r2_hi = bootstrap_ci(y_test_full.values, pred_test, r2_score)
            rows.append({
                "stage": stage_name, "model_type": model_type,
                "r2": r2, "r2_lo": r2_lo, "r2_hi": r2_hi,
            })
            print(f"{stage_name:30s} {model_type:11s} R2={r2:.3f}  95% CI=[{r2_lo:.3f}, {r2_hi:.3f}]")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(OUT_DIR, "regression_results_healthy_prediabetic.csv"), index=False)
    make_r2_forest_plot(results_df, os.path.join(OUT_DIR, "insulin_r2_forest_healthy_prediabetic.png"))


if __name__ == "__main__":
    main()
