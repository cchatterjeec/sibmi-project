"""
ROC curve for the best-performing model at Model 6a (Age/Sex/Race + BMI +
Wearables + CGM Metrics, no routine labs) using the curated/VIF-pruned CGM
feature set (21 features), for the direct study_group classification
(healthy vs pre-diabetic) -- run_classification_no_correlated_features.py.

"Best performing" = higher test-set AUROC between ElasticNet and XGBoost
at this stage (XGBoost: 0.682 vs ElasticNet: 0.675, per
classification_results_no_correlated_features.csv) -- both are fit here
and the better one is plotted.

Output: roc_curve_model6a_curated.png
"""
import os
import sys

import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablation_common import (
    HBA1C_COL,
    MODEL_TYPE_STYLE,
    fit_elasticnet_classifier,
    fit_xgb_classifier,
    get_ablation_stages,
    load_data,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "regression_no_correlated_features"))
from select_cgm_features_vif import CGM_FEATURES_VIF_PRUNED

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
NEG_LABEL = "healthy"
POS_LABEL = "pre_diabetes_lifestyle_controlled"

df = load_data()
df = df[df["study_group"].isin([NEG_LABEL, POS_LABEL])].copy()
df["y"] = (df["study_group"] == POS_LABEL).astype(int)
train = df[df["split"] == "train"]
test = df[df["split"] == "test"]
y_train, y_test = train["y"], test["y"].values

stages = get_ablation_stages(
    df, exclude_cpeptide=False, include_no_labs_variants=True,
    target_col=HBA1C_COL, cgm_features=CGM_FEATURES_VIF_PRUNED,
)
stage_name, feats = [s for s in stages if s[0].startswith("Model 6a")][0]
X_train, X_test = train[feats], test[feats]

fitters = {"ElasticNet": fit_elasticnet_classifier, "XGBoost": fit_xgb_classifier}
results = {}
for model_name, fitter in fitters.items():
    model = fitter(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, proba)
    fpr, tpr, _ = roc_curve(y_test, proba)
    results[model_name] = (fpr, tpr, auroc)
    print(f"{model_name}: AUROC={auroc:.3f}")

best_model = max(results, key=lambda m: results[m][2])
fpr, tpr, auroc = results[best_model]
print(f"best performing: {best_model} (AUROC={auroc:.3f})")

fig, ax = plt.subplots(figsize=(6.5, 6), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
ax.plot([0, 1], [0, 1], color="#999999", linewidth=1.0, linestyle="--", zorder=1, label="Chance")
ax.plot(
    fpr, tpr, color=MODEL_TYPE_STYLE[best_model]["color"], linewidth=2.0,
    zorder=3, label=f"{best_model} (AUROC = {auroc:.3f})",
)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("False Positive Rate", fontsize=11, color="#0b0b0b")
ax.set_ylabel("True Positive Rate", fontsize=11, color="#0b0b0b")
ax.set_title(
    f"ROC curve: {stage_name}\nhealthy vs pre-diabetic, CGM: no correlated features",
    fontsize=12, color="#0b0b0b", pad=12,
)
ax.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.legend(loc="lower right", frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "roc_curve_model6a_curated.png"), dpi=200, facecolor=fig.get_facecolor())
print("saved roc_curve_model6a_curated.png")
