"""
Classification ablation: predict healthy (0) vs pre-diabetic (1) --
study_group directly, cohort restricted to just these two groups (excludes
oral-medication and insulin-dependent participants entirely) -- using
ElasticNet (logistic, elastic-net penalty) and XGBoost across all 8
ablation stages, with the FULL CGM ML feature catalog (67 features) minus
gmi_pct as the CGM block.

gmi_pct is excluded for the same reason as the HbA1c regression scripts in
regression_all_features/: it's an algebraic estimated-A1c formula, and
since study_group here is clinically HbA1c-based, gmi_pct would be a
near-tautological predictor. HbA1c itself (import_hba1c) is also excluded
from the routine-labs block (Model 4) via target_col, for the same reason
-- otherwise "Model 4: + Routine Labs" mostly just measures whether the
model can read HbA1c off the labs sheet.

Outputs (in this directory):
  classification_results_all_features.csv
  classification_auroc_forest_all_features.png
"""
import os
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablation_common import (
    CGM_ML_FEATURES,
    HBA1C_COL,
    bootstrap_ci,
    fit_elasticnet_classifier,
    fit_xgb_classifier,
    get_ablation_stages,
    load_data,
    make_auroc_forest_plot,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
POS_LABEL = "pre_diabetes_lifestyle_controlled"  # positive class = pre-diabetic
NEG_LABEL = "healthy"
CGM_FEATURES_NO_GMI = [c for c in CGM_ML_FEATURES if c != "gmi_pct"]


def main():
    df = load_data()
    df = df[df["study_group"].isin([NEG_LABEL, POS_LABEL])].copy()
    df["y"] = (df["study_group"] == POS_LABEL).astype(int)
    print(f"cohort: n={len(df)}  ({(df['y']==0).sum()} healthy, {(df['y']==1).sum()} pre-diabetic)")

    stages = get_ablation_stages(
        df, exclude_cpeptide=False, include_no_labs_variants=True,
        target_col=HBA1C_COL, cgm_features=CGM_FEATURES_NO_GMI,
    )

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    y_train, y_test = train["y"], test["y"].values

    fitters = {"ElasticNet": fit_elasticnet_classifier, "XGBoost": fit_xgb_classifier}

    rows = []
    for stage_name, feats in stages:
        X_train, X_test = train[feats], test[feats]
        for model_type, fitter in fitters.items():
            model = fitter(X_train, y_train)
            proba_test = model.predict_proba(X_test)[:, 1]
            auroc = roc_auc_score(y_test, proba_test)
            auroc_lo, auroc_hi = bootstrap_ci(y_test, proba_test, roc_auc_score)
            rows.append({
                "stage": stage_name, "model_type": model_type,
                "auroc": auroc, "auroc_lo": auroc_lo, "auroc_hi": auroc_hi,
            })
            print(f"{stage_name:40s} {model_type:11s} AUROC={auroc:.3f}  95% CI=[{auroc_lo:.3f}, {auroc_hi:.3f}]")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(OUT_DIR, "classification_results_all_features.csv"), index=False)
    make_auroc_forest_plot(
        results_df, os.path.join(OUT_DIR, "classification_auroc_forest_all_features.png"),
        title="Healthy vs pre-diabetic AUROC across ablation stages\n(CGM: all features except gmi_pct)",
    )


if __name__ == "__main__":
    main()
