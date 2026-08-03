"""
Confusion matrices for the HbA1c-threshold-derived classification
(healthy vs pre-diabetic, cohort restricted to study_group in {healthy,
pre_diabetes_lifestyle_controlled}) at Model 6a (Age/Sex/Race + BMI +
Wearables + CGM Metrics, no routine labs) -- the regressor's continuous
HbA1c prediction is thresholded at 5.7 to get the predicted class.

One combined 2x2 figure: rows = CGM feature-set variant (all 66 features
minus gmi_pct, vs the 21-feature VIF-pruned/curated set), columns = model
type (ElasticNet, XGBoost).

Output: hba1c_threshold_classification_confusion_matrices.png
"""
import os
import sys

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablation_common import (
    CGM_ML_FEATURES,
    HBA1C_COL,
    fit_elasticnet_regressor,
    fit_xgb_regressor,
    get_ablation_stages,
    load_data,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "regression_no_correlated_features"))
from select_cgm_features_vif import CGM_FEATURES_VIF_PRUNED

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
NEG_LABEL = "healthy"
POS_LABEL = "pre_diabetes_lifestyle_controlled"
CGM_FEATURES_NO_GMI = [c for c in CGM_ML_FEATURES if c != "gmi_pct"]
THRESHOLD = 5.7

df = load_data()
df = df[df["study_group"].isin([NEG_LABEL, POS_LABEL])].copy()
train = df[df["split"] == "train"]
test = df[df["split"] == "test"]
y_train_full, y_test_full = train[HBA1C_COL], test[HBA1C_COL]
y_test_class = (y_test_full.values >= THRESHOLD).astype(int)

variants = {
    "All CGM features (66)": CGM_FEATURES_NO_GMI,
    "Curated CGM features (21)": CGM_FEATURES_VIF_PRUNED,
}
fitters = {"ElasticNet": fit_elasticnet_regressor, "XGBoost": fit_xgb_regressor}

fig, axes = plt.subplots(2, 2, figsize=(9, 8.5), facecolor="#fcfcfb")

for row, (variant_name, cgm_feats) in enumerate(variants.items()):
    stages = get_ablation_stages(
        df, exclude_cpeptide=True, include_no_labs_variants=True,
        target_col=HBA1C_COL, cgm_features=cgm_feats,
    )
    stage_name, feats = [s for s in stages if s[0].startswith("Model 6a")][0]
    X_train, X_test = train[feats], test[feats]

    for col, (model_name, fitter) in enumerate(fitters.items()):
        model = fitter(X_train, y_train_full)
        pred_test = model.predict(X_test)
        y_pred_class = (pred_test >= THRESHOLD).astype(int)
        cm = confusion_matrix(y_test_class, y_pred_class, labels=[0, 1])

        ax = axes[row, col]
        ax.imshow(cm, cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        ax.set_xticks([0, 1]); ax.set_xticklabels(["healthy", "pre-diabetic"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["healthy", "pre-diabetic"])
        ax.set_xlabel("Predicted (HbA1c thresh. 5.7)")
        ax.set_ylabel("Actual (HbA1c thresh. 5.7)")
        ax.set_title(f"{variant_name}\n{model_name} (Model 6a)", fontsize=11)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "hba1c_threshold_classification_confusion_matrices.png"), dpi=200, facecolor=fig.get_facecolor())
print("saved hba1c_threshold_classification_confusion_matrices.png")
