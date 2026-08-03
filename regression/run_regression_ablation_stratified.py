"""
Same insulin regression ablation as run_regression_ablation.py (ElasticNet
and XGBoost across all ablation stages, c-peptide excluded throughout), but
evaluation is broken down by diabetes status (study_group) instead of pooled
across the whole test set. Each model is fit once per stage on the full
train split; every study_group's test-set rows are then scored separately,
producing one R2-vs-stage forest plot per study_group (4 total) instead of
one plot pooling everyone together.

Outputs (in this directory):
  regression_results_stratified.csv       -- point R2 + 95% bootstrap CIs, per stage/model/study_group
  insulin_r2_forest_<study_group>.png     -- one forest plot per study_group
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


def main():
    df = load_data()
    stages = get_ablation_stages(df, exclude_cpeptide=True, include_no_labs_variants=True)

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    y_train, y_test = train[INSULIN_COL], test[INSULIN_COL].values

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
    results_df.to_csv(os.path.join(OUT_DIR, "regression_results_stratified.csv"), index=False)

    for study_group in study_groups:
        sub = results_df[results_df["study_group"] == study_group]
        make_r2_forest_plot(
            sub, os.path.join(OUT_DIR, f"insulin_r2_forest_{study_group}.png"),
            title=f"Insulin regression $R^2$ across ablation stages\n({study_group})",
        )


if __name__ == "__main__":
    main()
