"""
HbA1c regression ablation, stratified by study_group (diabetes status),
using the VIF-pruned CGM feature set (select_cgm_features_vif.py -- 23 of
the 67 catalog features survive VIF <= 10, computed on the CGM block alone;
nothing else in the model -- demographics, wearables, routine labs -- is
pruned). c-peptide is excluded throughout (same leakage rationale as
run_hba1c_regression_ablation.py); insulin is never used as a predictor
(excluded by get_ablation_stages); HbA1c itself is excluded from the
routine-labs feature set via target_col.

Same design as run_insulin_regression_stratified.py: each stage is fit once
on the full train split, then every study_group's test-set rows are scored
separately, producing one R2-vs-stage forest plot per study_group.

Outputs (in this directory):
  hba1c_regression_results_stratified.csv -- point R2 + 95% bootstrap CIs, per stage/model/study_group
  hba1c_r2_forest_<study_group>.png       -- one forest plot per study_group
"""
import os
import sys

import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ablation_common import (
    HBA1C_COL,
    bootstrap_ci,
    fit_elasticnet_regressor,
    fit_xgb_regressor,
    get_ablation_stages,
    load_data,
    make_r2_forest_plot,
)
from select_cgm_features_vif import CGM_FEATURES_VIF_PRUNED

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    df = load_data()
    stages = get_ablation_stages(
        df, exclude_cpeptide=True, include_no_labs_variants=True, target_col=HBA1C_COL,
        cgm_features=CGM_FEATURES_VIF_PRUNED,
    )

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    y_train, y_test = train[HBA1C_COL], test[HBA1C_COL].values

    fitters = {"ElasticNet": fit_elasticnet_regressor, "XGBoost": fit_xgb_regressor}

    predictions = {}
    for stage_name, feats in stages:
        X_train, X_test = train[feats], test[feats]
        for model_type, fitter in fitters.items():
            model = fitter(X_train, y_train)
            predictions[(stage_name, model_type)] = model.predict(X_test)

    study_groups = sorted(test["study_group"].unique())
    rows = []
    for study_group in study_groups:
        mask = (test["study_group"] == study_group).values
        y_true = y_test[mask]
        for stage_name, feats in stages:
            for model_type in fitters:
                y_pred = predictions[(stage_name, model_type)][mask]
                r2 = r2_score(y_true, y_pred)
                r2_lo, r2_hi = bootstrap_ci(y_true, y_pred, r2_score)
                rows.append({
                    "study_group": study_group, "stage": stage_name, "model_type": model_type,
                    "n": int(mask.sum()), "r2": r2, "r2_lo": r2_lo, "r2_hi": r2_hi,
                })
                print(f"{study_group:70s} {stage_name:30s} {model_type:11s} "
                      f"n={mask.sum():4d}  R2={r2:.3f}  95% CI=[{r2_lo:.3f}, {r2_hi:.3f}]")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(OUT_DIR, "hba1c_regression_results_stratified.csv"), index=False)

    dropped_stages = {"Model 1: Age, Sex, Race", "Model 2: + BMI", "Model 3: + Wearables", "Model 4: + Routine Labs"}
    for study_group in study_groups:
        sub = results_df[(results_df["study_group"] == study_group) & (~results_df["stage"].isin(dropped_stages))]
        make_r2_forest_plot(
            sub, os.path.join(OUT_DIR, f"hba1c_r2_forest_{study_group}.png"),
            title=f"HbA1c regression $R^2$ across ablation stages\n({study_group})",
            xlim=(0, 1),
        )


if __name__ == "__main__":
    main()
