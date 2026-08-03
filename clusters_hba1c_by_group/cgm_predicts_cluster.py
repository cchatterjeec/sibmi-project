"""
Can CGM alone (no gmi_pct) recover the age/BMI/insulin/C-peptide/HbA1c
cluster assignment within each study group?

Positive class = cluster b (cluster 1 -- the higher adiposity/insulin/
C-peptide group per clustering.py's canonicalization). gmi_pct is excluded
from the CGM feature block because it's an algebraic eA1c estimate
(3.31 + 0.02392*mean_glucose) and the clusters are HbA1c-informed, so
leaving it in would make this a near-tautological prediction rather than a
genuine test of whether the CGM *signal* (variability, dynamics, complexity,
etc.) carries the phenotype -- same reasoning as classification/run_classification_*.py.

Two CGM feature-set variants are run, same as classification/'s
all_features / no_correlated_features split:
  all_features          -- full 66-feature CGM catalog minus gmi_pct
  no_correlated_features -- the 20-feature VIF-pruned/curated set
                            (regression_no_correlated_features/select_cgm_features_vif.py),
                            which already excludes gmi_pct

Per group: fit on the existing train split, evaluate on the held-out test
split (final_df.csv "split" column), with participants gated on CGM QC
(qc_pct_active >= 70%, via ablation_common.load_data) and restricted to
those with a cluster assignment for that group. ElasticNet and XGBoost are
both fit, mirroring the rest of the ablation pipeline's convention.

Metrics (bootstrap 95% CI via resampling test rows, ablation_common.bootstrap_ci):
  AUPRC       -- average_precision_score(y_test, proba_test)
  sensitivity -- recall for cluster b at the 0.5 probability threshold
  specificity -- recall for cluster a (the negative class) at 0.5

Outputs (in clusters_hba1c_by_group/), one set per feature-set variant:
  cgm_predicts_cluster_results_{variant}.csv
  cgm_predicts_cluster_auprc_forest_{variant}.png       -- x in [0,1], per-group prevalence tick
  cgm_predicts_cluster_sensitivity_forest_{variant}.png -- x in [0,1]
  cgm_predicts_cluster_specificity_forest_{variant}.png -- x in [0,1]
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "classification"))
from ablation_common import (
    CGM_ML_FEATURES,
    MODEL_TYPE_STYLE,
    bootstrap_ci,
    fit_elasticnet_classifier,
    fit_xgb_classifier,
    load_data,
)

sys.path.insert(0, os.path.join(ROOT, "regression_no_correlated_features"))
from select_cgm_features_vif import CGM_FEATURES_VIF_PRUNED

import importlib.util

# See avg_linkage_distance.py for why this isn't a bare `import common` --
# clusters_9_vars_by_group/ has its own unrelated common.py that can silently
# shadow this one if imported first in the same process.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
_COMMON_KEY = f"common__{os.path.basename(_THIS_DIR)}"
if _COMMON_KEY in sys.modules:
    C = sys.modules[_COMMON_KEY]
else:
    _spec = importlib.util.spec_from_file_location(_COMMON_KEY, os.path.join(_THIS_DIR, "common.py"))
    C = importlib.util.module_from_spec(_spec)
    sys.modules[_COMMON_KEY] = C
    _spec.loader.exec_module(C)

OUT_DIR = C.OUT_ROOT
FEATURE_SETS = {
    "all_features": [c for c in CGM_ML_FEATURES if c != "gmi_pct"],
    "no_correlated_features": CGM_FEATURES_VIF_PRUNED,
}
GROUP_LABELS = {
    "healthy": "Healthy",
    "prediabetic": "Prediabetic",
    "oral_medication": "Oral medication",
    "insulin_dependent": "Insulin dependent",
}
FITTERS = {"ElasticNet": fit_elasticnet_classifier, "XGBoost": fit_xgb_classifier}


def sensitivity_score(y_true, y_score, threshold=0.5):
    y_true, y_pred = np.asarray(y_true), (np.asarray(y_score) >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp / (tp + fn) if (tp + fn) else float("nan")


def specificity_score(y_true, y_score, threshold=0.5):
    y_true, y_pred = np.asarray(y_true), (np.asarray(y_score) >= threshold).astype(int)
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return tn / (tn + fp) if (tn + fp) else float("nan")


def run_group(slug, df, feats):
    idc = C.id_col(df)
    assign = pd.read_csv(os.path.join(C.out_dir(slug), "cluster_assignments.csv"))
    merged = df.merge(assign[[idc, "cluster"]], on=idc, how="inner")
    merged["y"] = merged["cluster"].astype(int)  # cluster b (1) = positive

    train = merged[merged["split"] == "train"]
    test = merged[merged["split"] == "test"]
    X_train, X_test = train[feats], test[feats]
    y_train, y_test = train["y"].values, test["y"].values
    prevalence = float(y_test.mean())
    print(f"[{slug}] n_train={len(train)} n_test={len(test)} "
          f"(test: {int((y_test==1).sum())} cluster-b / {len(y_test)}, prevalence={prevalence:.3f})")

    rows = []
    for model_type, fitter in FITTERS.items():
        model = fitter(X_train, y_train)
        proba_test = model.predict_proba(X_test)[:, 1]

        auprc = average_precision_score(y_test, proba_test)
        auprc_lo, auprc_hi = bootstrap_ci(y_test, proba_test, average_precision_score)
        sens = sensitivity_score(y_test, proba_test)
        sens_lo, sens_hi = bootstrap_ci(y_test, proba_test, sensitivity_score)
        spec = specificity_score(y_test, proba_test)
        spec_lo, spec_hi = bootstrap_ci(y_test, proba_test, specificity_score)

        rows.append({
            "group": slug, "model_type": model_type, "n_test": len(test), "prevalence": prevalence,
            "auprc": auprc, "auprc_lo": auprc_lo, "auprc_hi": auprc_hi,
            "sensitivity": sens, "sensitivity_lo": sens_lo, "sensitivity_hi": sens_hi,
            "specificity": spec, "specificity_lo": spec_lo, "specificity_hi": spec_hi,
        })
        print(f"  {model_type:11s} AUPRC={auprc:.3f} [{auprc_lo:.3f},{auprc_hi:.3f}]  "
              f"sens={sens:.3f} [{sens_lo:.3f},{sens_hi:.3f}]  "
              f"spec={spec:.3f} [{spec_lo:.3f},{spec_hi:.3f}]")
    return rows


def make_forest_plot(results_df, out_path, metric, title, show_prevalence=False):
    groups = list(GROUP_LABELS)
    model_types = list(MODEL_TYPE_STYLE.keys())
    dodge = {model_types[0]: 0.12, model_types[1]: -0.12}
    n_groups = len(groups)
    y_base = np.arange(n_groups)[::-1]

    fig, ax = plt.subplots(figsize=(9.5, 1.1 * n_groups + 1.8), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    if show_prevalence:
        prev_by_group = results_df.drop_duplicates("group").set_index("group")["prevalence"]
        for gi, g in enumerate(groups):
            y = y_base[gi]
            ax.plot(
                [prev_by_group[g]] * 2, [y - 0.32, y + 0.32],
                color="#999999", linewidth=1.6, linestyle="--", zorder=2,
                label="cluster-b prevalence (no-skill baseline)" if gi == 0 else None,
            )

    for mt in model_types:
        sub = results_df[results_df["model_type"] == mt].set_index("group").loc[groups]
        y = y_base + dodge[mt]
        x = sub[metric].values
        lo = x - sub[f"{metric}_lo"].values
        hi = sub[f"{metric}_hi"].values - x
        container = ax.errorbar(
            x, y, xerr=[lo, hi], fmt=MODEL_TYPE_STYLE[mt]["marker"],
            color=MODEL_TYPE_STYLE[mt]["color"], ecolor=MODEL_TYPE_STYLE[mt]["color"],
            elinewidth=1.4, capsize=3, markersize=9, markeredgecolor="#0b0b0b",
            markeredgewidth=0.6, label=mt, zorder=3, clip_on=False,
        )
        # Points/CIs can sit exactly at 0 or 1 (e.g. a perfect-specificity fold) --
        # clip_on=False keeps those markers from being half-cut at the axis edge
        # while xlim below still hard-cuts the visible range at [0, 1].
        data_line, caplines, barlinecols = container.lines
        if data_line is not None:
            data_line.set_clip_on(False)
        for cap in caplines:
            cap.set_clip_on(False)
        for col in barlinecols:
            col.set_clip_on(False)

    n_by_group = results_df.drop_duplicates("group").set_index("group")["n_test"]
    labels = [f"{GROUP_LABELS[g]} (n={n_by_group[g]})" for g in groups]
    ax.set_yticks(y_base)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_xlabel(title["xlabel"], fontsize=11, color="#0b0b0b")
    ax.set_title(title["title"], fontsize=11.5, color="#0b0b0b", pad=12)
    ax.xaxis.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="upper center", bbox_to_anchor=(0.58, 1.0),
        ncol=len(handles), frameon=False, fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    df = load_data()
    for variant, feats in FEATURE_SETS.items():
        print(f"=== feature set: {variant} ({len(feats)} CGM features) ===")
        all_rows = []
        for slug in GROUP_LABELS:
            all_rows.extend(run_group(slug, df, feats))
        results_df = pd.DataFrame(all_rows)
        csv_path = os.path.join(OUT_DIR, f"cgm_predicts_cluster_results_{variant}.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"saved {csv_path}")

        variant_label = "all CGM features" if variant == "all_features" else "curated CGM features"
        make_forest_plot(
            results_df, os.path.join(OUT_DIR, f"cgm_predicts_cluster_auprc_forest_{variant}.png"), "auprc",
            {"title": f"CGM-only prediction of cluster b, by study group ({variant_label})\n"
                      "AUPRC (cluster b = positive class), 95% CI",
             "xlabel": "AUPRC"},
            show_prevalence=True,
        )
        make_forest_plot(
            results_df, os.path.join(OUT_DIR, f"cgm_predicts_cluster_sensitivity_forest_{variant}.png"),
            "sensitivity",
            {"title": f"CGM-only prediction of cluster b, by study group ({variant_label})\n"
                      "Sensitivity at p>=0.5, 95% CI",
             "xlabel": "Sensitivity (cluster b recall)"},
        )
        make_forest_plot(
            results_df, os.path.join(OUT_DIR, f"cgm_predicts_cluster_specificity_forest_{variant}.png"),
            "specificity",
            {"title": f"CGM-only prediction of cluster b, by study group ({variant_label})\n"
                      "Specificity at p>=0.5, 95% CI",
             "xlabel": "Specificity (cluster a recall)"},
        )


if __name__ == "__main__":
    main()
