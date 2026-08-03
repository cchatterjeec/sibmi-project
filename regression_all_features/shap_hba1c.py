"""
SHAP beeswarm plots for the HbA1c Model 6a fit (demographics + BMI +
wearables + full CGM ML feature catalog minus gmi_pct, no routine labs) --
the same stage and same gmi_pct exclusion as run_hba1c_regression_ablation.py
/ eval_model_6a_stratified.py in this directory. Models are refit here
(fit_elasticnet_regressor / fit_xgb_regressor are deterministic given
random_state=0, so these are the same fits used elsewhere in this
directory) purely to get SHAP explainers against them.

ElasticNet is linear, so SHAP is exact via shap.LinearExplainer on the
pipeline's standardized feature space. XGBoost is exact via
shap.TreeExplainer on the raw (unscaled) feature space.

Output: one combined PNG (both model types as two panels) --
  hba1c_shap_model6a.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ablation_common import (
    CGM_ML_FEATURES,
    HBA1C_COL,
    fit_elasticnet_regressor,
    fit_xgb_regressor,
    get_ablation_stages,
    load_data,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE_NAME = "Model 6a: Model 3 + CGM (no labs)"
MAX_DISPLAY = 20
CGM_FEATURES_NO_GMI = [c for c in CGM_ML_FEATURES if c != "gmi_pct"]


def main():
    df = load_data()
    stages = dict(get_ablation_stages(
        df, exclude_cpeptide=True, include_no_labs_variants=True, target_col=HBA1C_COL,
        cgm_features=CGM_FEATURES_NO_GMI,
    ))
    feats = stages[STAGE_NAME]

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    X_train, y_train = train[feats], train[HBA1C_COL]
    X_test = test[feats]

    print(f"[shap] fitting on {len(feats)} Model 6a features, n_train={len(X_train)}, n_test={len(X_test)}")

    enet = fit_elasticnet_regressor(X_train, y_train)
    xgb = fit_xgb_regressor(X_train, y_train)

    scaler, reg = enet.named_steps["scaler"], enet.named_steps["reg"]
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    explainer_enet = shap.LinearExplainer(reg, X_train_scaled, feature_names=feats)
    sv_enet = explainer_enet(X_test_scaled)

    explainer_xgb = shap.TreeExplainer(xgb)
    sv_xgb = explainer_xgb(X_test)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 0.35 * MAX_DISPLAY + 3))
    plt.sca(ax1)
    shap.plots.beeswarm(sv_enet, max_display=MAX_DISPLAY, ax=ax1, show=False, plot_size=None)
    ax1.set_title(f"ElasticNet -- Model 6a HbA1c SHAP\n({len(feats)} features, all_features directory)", fontsize=11)

    plt.sca(ax2)
    shap.plots.beeswarm(sv_xgb, max_display=MAX_DISPLAY, ax=ax2, show=False, plot_size=None)
    ax2.set_title(f"XGBoost -- Model 6a HbA1c SHAP\n({len(feats)} features, all_features directory)", fontsize=11)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "hba1c_shap_model6a.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OUT] wrote {out_path}")


if __name__ == "__main__":
    main()
