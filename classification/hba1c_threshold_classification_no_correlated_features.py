"""
Threshold-derived classification: healthy (0) vs pre-diabetic (1), cohort
restricted to study_group in {healthy, pre_diabetes_lifestyle_controlled}
(same cohort as run_classification_no_correlated_features.py in this
directory) -- but here the label comes from thresholding a regressor's
continuous HbA1c prediction at 5.7 (healthy < 5.7 <= pre-diabetic, per the
ADA prediabetes cutoff), instead of training a classifier on study_group
directly.

Same ablation stages, ElasticNet/XGBoost regressors, and VIF-pruned/
curated CGM feature set (regression_no_correlated_features/
select_cgm_features_vif.py -- 21 features) as
regression_no_correlated_features/run_hba1c_regression_ablation.py --
refit here on the restricted cohort (not reused from that script's
full-population fit).

AUROC is computed using the continuous predicted HbA1c as the score; the
5.7 threshold is only applied to get a hard predicted class for the
confusion matrix.

This is a distinct analysis from run_classification_no_correlated_features.py
in this directory, which trains a classifier directly on the study_group
label. Here the label is *derived* post-hoc from a regressor's continuous
HbA1c prediction -- see conversation notes: this tends to score higher
than the direct classification because CGM mean_glucose is a near-direct
physiological proxy for HbA1c itself (same relationship gmi_pct makes
explicit), so thresholding a good HbA1c regression is an easier task than
predicting the clinical study_group label from CGM.

Confusion matrices for this analysis live in the separate combined script
hba1c_threshold_classification_confusion_matrices.py (Model 6a, both CGM
variants, both models, in one plot).

Outputs (in this directory):
  hba1c_threshold_classification_results_no_correlated_features.csv
  hba1c_threshold_classification_auroc_forest_no_correlated_features.png
"""
import os
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablation_common import (
    HBA1C_COL,
    bootstrap_ci,
    fit_elasticnet_regressor,
    fit_xgb_regressor,
    get_ablation_stages,
    load_data,
    make_auroc_forest_plot,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "regression_no_correlated_features"))
from select_cgm_features_vif import CGM_FEATURES_VIF_PRUNED

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
NEG_LABEL = "healthy"
POS_LABEL = "pre_diabetes_lifestyle_controlled"
THRESHOLD = 5.7


def main():
    df = load_data()
    df = df[df["study_group"].isin([NEG_LABEL, POS_LABEL])].copy()
    print(f"cohort: n={len(df)}  ({(df['study_group']==NEG_LABEL).sum()} healthy, "
          f"{(df['study_group']==POS_LABEL).sum()} pre-diabetic)")

    stages = get_ablation_stages(
        df, exclude_cpeptide=True, include_no_labs_variants=True, target_col=HBA1C_COL,
        cgm_features=CGM_FEATURES_VIF_PRUNED,
    )

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    y_train_full, y_test_full = train[HBA1C_COL], test[HBA1C_COL]
    y_test_class = (y_test_full.values >= THRESHOLD).astype(int)

    fitters = {"ElasticNet": fit_elasticnet_regressor, "XGBoost": fit_xgb_regressor}

    rows = []
    for stage_name, feats in stages:
        X_train, X_test = train[feats], test[feats]
        for model_type, fitter in fitters.items():
            model = fitter(X_train, y_train_full)
            pred_test = model.predict(X_test)
            auroc = roc_auc_score(y_test_class, pred_test)
            auroc_lo, auroc_hi = bootstrap_ci(y_test_class, pred_test, roc_auc_score)
            rows.append({
                "stage": stage_name, "model_type": model_type,
                "auroc": auroc, "auroc_lo": auroc_lo, "auroc_hi": auroc_hi,
            })
            print(f"{stage_name:40s} {model_type:11s} AUROC={auroc:.3f}  95% CI=[{auroc_lo:.3f}, {auroc_hi:.3f}]")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(OUT_DIR, "hba1c_threshold_classification_results_no_correlated_features.csv"), index=False)
    make_auroc_forest_plot(
        results_df, os.path.join(OUT_DIR, "hba1c_threshold_classification_auroc_forest_no_correlated_features.png"),
        title="HbA1c-threshold (5.7) AUROC across ablation stages\n(CGM: no correlated features)",
    )


if __name__ == "__main__":
    main()
