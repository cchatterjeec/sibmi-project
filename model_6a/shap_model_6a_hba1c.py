"""
SHAP feature attributions for Model 6a (demographics + BMI + wearables + CGM
metrics, routine labs skipped) predicting HbA1c, for both ElasticNet and
XGBoost. Models are fit on the train split and explained on the test split,
matching the evaluation split used in run_hba1c_regression_ablation.py.

Outputs (in this directory):
  model_6a_hba1c_shap_elasticnet.png
  model_6a_hba1c_shap_xgboost.png
  model_6a_hba1c_shap_mean_abs.csv -- mean |SHAP| per feature per model
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ablation_common import (
    HBA1C_COL,
    fit_elasticnet_regressor,
    fit_xgb_regressor,
    get_ablation_stages,
    load_data,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE_NAME = "Model 6a: Model 3 + CGM (no labs)"

df = load_data()
stages = dict(get_ablation_stages(
    df, exclude_cpeptide=True, include_no_labs_variants=True, target_col=HBA1C_COL,
))
feats = stages[STAGE_NAME]

train = df[df["split"] == "train"]
test = df[df["split"] == "test"]
X_train, X_test = train[feats], test[feats]
y_train = train[HBA1C_COL]

enet = fit_elasticnet_regressor(X_train, y_train)
xgb = fit_xgb_regressor(X_train, y_train)

scaler = enet.named_steps["scaler"]
reg = enet.named_steps["reg"]
explainer_enet = shap.LinearExplainer(reg, scaler.transform(X_train))
shap_values_enet = explainer_enet(scaler.transform(X_test))
shap_values_enet.feature_names = feats

explainer_xgb = shap.TreeExplainer(xgb)
shap_values_xgb = explainer_xgb(X_test)

mean_abs = pd.DataFrame({
    "feature": feats,
    "elasticnet_mean_abs_shap": np.abs(shap_values_enet.values).mean(axis=0),
    "xgboost_mean_abs_shap": np.abs(shap_values_xgb.values).mean(axis=0),
}).sort_values("xgboost_mean_abs_shap", ascending=False)
mean_abs.to_csv(os.path.join(OUT_DIR, "model_6a_hba1c_shap_mean_abs.csv"), index=False)
print(mean_abs.to_string(index=False))

for shap_values, model_name in [(shap_values_enet, "elasticnet"), (shap_values_xgb, "xgboost")]:
    plt.figure(figsize=(8, 0.35 * len(feats) + 2), facecolor="#fcfcfb")
    shap.plots.beeswarm(shap_values, show=False)
    plt.title(f"{model_name.title()} SHAP values -- Model 6a (HbA1c)", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, f"model_6a_hba1c_shap_{model_name}.png"),
        dpi=200, facecolor="#fcfcfb",
    )
    plt.close()
