"""
Regression ablation: predict serum insulin using ElasticNet and XGBoost
across cumulative feature-ablation stages, plus two variants that add CGM
metrics or serum glucose directly on top of demographics + BMI + wearables
(skipping routine labs). c-peptide is excluded from the feature set
throughout (co-secreted with insulin -> leakage).

Outputs (in this directory):
  regression_results.csv  -- point R2 estimates + 95% bootstrap CIs
  insulin_r2_forest.png   -- forest plot, R2 on x-axis, ablation stage on y-axis
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ablation_common import (
    INSULIN_COL,
    MODEL_TYPE_STYLE,
    bootstrap_ci,
    fit_elasticnet_regressor,
    fit_xgb_regressor,
    get_ablation_stages,
    load_data,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def make_r2_forest_plot(results_df, out_path, title="Insulin regression $R^2$ across ablation stages"):
    stages = results_df["stage"].unique().tolist()
    model_types = list(MODEL_TYPE_STYLE.keys())
    n_stages = len(stages)
    y_base = np.arange(n_stages)[::-1]  # first stage at top
    dodge = {model_types[0]: 0.12, model_types[1]: -0.12}

    fig, ax = plt.subplots(figsize=(8, 1.05 * n_stages + 1.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for mt in model_types:
        sub = results_df[results_df["model_type"] == mt].set_index("stage").loc[stages]
        y = y_base + dodge[mt]
        x = sub["r2"].values
        lo = x - sub["r2_lo"].values
        hi = sub["r2_hi"].values - x
        ax.errorbar(
            x, y, xerr=[lo, hi], fmt=MODEL_TYPE_STYLE[mt]["marker"],
            color=MODEL_TYPE_STYLE[mt]["color"], ecolor=MODEL_TYPE_STYLE[mt]["color"],
            elinewidth=1.4, capsize=3, markersize=9, markeredgecolor="#0b0b0b",
            markeredgewidth=0.6, label=mt, zorder=3,
        )

    ax.set_yticks(y_base)
    ax.set_yticklabels(stages, fontsize=10)
    ax.set_xlabel(r"$R^2$", fontsize=11, color="#0b0b0b")
    ax.set_title(title, fontsize=13, color="#0b0b0b", pad=12)
    ax.xaxis.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    df = load_data()
    stages = get_ablation_stages(df, exclude_cpeptide=True, include_no_labs_variants=True)

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
    results_df.to_csv(os.path.join(OUT_DIR, "regression_results.csv"), index=False)
    make_r2_forest_plot(results_df, os.path.join(OUT_DIR, "insulin_r2_forest.png"))


if __name__ == "__main__":
    main()
