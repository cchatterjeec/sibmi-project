"""
Regression ablation: predict HbA1c using ElasticNet and XGBoost across
cumulative feature-ablation stages, using the FULL CGM ML feature catalog
(cgm_ml_features.py -- 67 features) as the Model 5a/6a CGM block, EXCEPT
gmi_pct.

gmi_pct is dropped here: it's an algebraic estimated-A1c formula
(3.31 + 0.02392*mean_glucose, Bergenstal 2018), so including it as a
predictor of lab HbA1c is near-tautological. mean_glucose itself is kept --
mean CGM glucose predicting HbA1c is genuine physiology (HbA1c reflects
~3-month average glycemic exposure), not leakage, and mean_glucose has no
special formulaic relationship to the target the way gmi_pct does.

Insulin is never used as a predictor (excluded by get_ablation_stages), and
c-peptide is excluded from the feature set throughout. HbA1c itself is
excluded from the routine-labs feature set via target_col to avoid leakage.

Outputs (in this directory):
  hba1c_regression_results.csv -- point R2 estimates + 95% bootstrap CIs
  hba1c_r2_forest.png          -- forest plot, R2 on x-axis, ablation stage on y-axis
"""
import os
import sys

import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ablation_common import (
    CGM_ML_FEATURES,
    HBA1C_COL,
    bootstrap_ci,
    fit_elasticnet_regressor,
    fit_xgb_regressor,
    get_ablation_stages,
    load_data,
    make_r2_forest_plot,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CGM_FEATURES_NO_GMI = [c for c in CGM_ML_FEATURES if c != "gmi_pct"]


def main():
    df = load_data()
    stages = get_ablation_stages(
        df, exclude_cpeptide=True, include_no_labs_variants=True, target_col=HBA1C_COL,
        cgm_features=CGM_FEATURES_NO_GMI,
    )

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    y_train_full, y_test_full = train[HBA1C_COL], test[HBA1C_COL]

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
    results_df.to_csv(os.path.join(OUT_DIR, "hba1c_regression_results.csv"), index=False)
    make_r2_forest_plot(
        results_df, os.path.join(OUT_DIR, "hba1c_r2_forest.png"),
        title="HbA1c regression $R^2$ across ablation stages",
    )


if __name__ == "__main__":
    main()
