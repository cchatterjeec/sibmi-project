"""
Break down Model 6a's HbA1c-prediction performance (demographics + BMI +
wearables + CGM metrics, no routine labs; ElasticNet and XGBoost) by
subgroup: sex, race, BMI category, and diabetes status (study_group).

The model is fit once on the full train split, exactly as in
run_hba1c_regression_ablation.py -- this script only slices the existing
test split by each stratifying variable and recomputes R2 within each
slice. No retraining per subgroup.

BMI categories are the weight_class labels already built in
stratification/strata_df.csv (Underweight <18.5, Normal 18.5-25,
Overweight 25-30, Obesity >=30 kg/m^2), joined onto final_df.csv by
participant_id.

Outputs (in this directory):
  hba1c_strat_sex.csv
  hba1c_strat_race.csv
  hba1c_strat_bmi_category.csv
  hba1c_strat_diabetes_status.csv
  hba1c_strat_forest.png -- forest plot, R2 per subgroup, one panel per stratifying variable
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ablation_common import (
    HBA1C_COL,
    MODEL_TYPE_STYLE,
    bootstrap_ci,
    fit_elasticnet_regressor,
    fit_xgb_regressor,
    get_ablation_stages,
    load_data,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE_NAME = "Model 6a: Model 3 + CGM (no labs)"
MIN_SUBGROUP_SIZE = 15

STRATA = {
    "sex": "sex",
    "race": "race",
    "bmi_category": "weight_class",
    "diabetes_status": "study_group",
}
BMI_CATEGORY_ORDER = ["Underweight", "Normal", "Overweight", "Obesity"]


def main():
    df = load_data()

    weight_class = pd.read_csv(
        os.path.join(OUT_DIR, "..", "stratification", "strata_df.csv"),
        usecols=["participant_id", "weight_class"],
    )
    df = df.merge(weight_class, on="participant_id", how="left")

    stages = dict(get_ablation_stages(
        df, exclude_cpeptide=True, include_no_labs_variants=True, target_col=HBA1C_COL,
    ))
    feats = stages[STAGE_NAME]

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"].copy()

    enet = fit_elasticnet_regressor(train[feats], train[HBA1C_COL])
    xgb = fit_xgb_regressor(train[feats], train[HBA1C_COL])
    test["pred_enet"] = enet.predict(test[feats])
    test["pred_xgb"] = xgb.predict(test[feats])

    all_results = {}
    for strat_name, col in STRATA.items():
        levels = test[col].dropna().unique().tolist()
        if strat_name == "bmi_category":
            levels = [lv for lv in BMI_CATEGORY_ORDER if lv in levels]
        else:
            levels = sorted(levels)

        rows = []
        for level in levels:
            sub = test[test[col] == level]
            if len(sub) < MIN_SUBGROUP_SIZE:
                print(f"Skipping {strat_name}={level}: n={len(sub)} < {MIN_SUBGROUP_SIZE}")
                continue
            y_true = sub[HBA1C_COL].values
            for model_name, pred_col in [("ElasticNet", "pred_enet"), ("XGBoost", "pred_xgb")]:
                y_pred = sub[pred_col].values
                r2 = r2_score(y_true, y_pred)
                r2_lo, r2_hi = bootstrap_ci(y_true, y_pred, r2_score)
                rows.append({
                    "level": level, "n": len(sub), "model_type": model_name,
                    "r2": r2, "r2_lo": r2_lo, "r2_hi": r2_hi,
                })
                print(f"{strat_name:18s} {str(level):45s} {model_name:11s} "
                      f"n={len(sub):4d}  R2={r2:.3f}  95% CI=[{r2_lo:.3f}, {r2_hi:.3f}]")

        results_df = pd.DataFrame(rows)
        results_df.to_csv(os.path.join(OUT_DIR, f"hba1c_strat_{strat_name}.csv"), index=False)
        all_results[strat_name] = results_df

    make_strat_forest_plot(all_results, os.path.join(OUT_DIR, "hba1c_strat_forest.png"))


def make_strat_forest_plot(all_results, out_path):
    model_types = list(MODEL_TYPE_STYLE.keys())
    dodge = {model_types[0]: 0.12, model_types[1]: -0.12}
    strat_names = list(all_results.keys())

    fig, axes = plt.subplots(
        len(strat_names), 1, figsize=(8, sum(1.0 * len(r) + 1.3 for r in all_results.values())),
        facecolor="#fcfcfb",
    )
    for ax, strat_name in zip(axes, strat_names):
        results_df = all_results[strat_name]
        ax.set_facecolor("#fcfcfb")
        levels = results_df["level"].unique().tolist()
        y_base = np.arange(len(levels))[::-1]

        for mt in model_types:
            sub = results_df[results_df["model_type"] == mt].set_index("level").loc[levels]
            y = y_base + dodge[mt]
            x = sub["r2"].values
            lo = x - sub["r2_lo"].values
            hi = sub["r2_hi"].values - x
            ax.errorbar(
                x, y, xerr=[lo, hi], fmt=MODEL_TYPE_STYLE[mt]["marker"],
                color=MODEL_TYPE_STYLE[mt]["color"], ecolor=MODEL_TYPE_STYLE[mt]["color"],
                elinewidth=1.4, capsize=3, markersize=8, markeredgecolor="#0b0b0b",
                markeredgewidth=0.6, label=mt, zorder=3,
            )

        n_by_level = results_df.drop_duplicates("level").set_index("level")["n"]
        labels = [f"{lv} (n={n_by_level[lv]})" for lv in levels]
        ax.set_yticks(y_base)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(r"$R^2$", fontsize=10)
        ax.set_title(f"Model 6a HbA1c $R^2$ by {strat_name.replace('_', ' ')}", fontsize=11, pad=8)
        ax.xaxis.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
