"""
Can CGM alone predict which of the 4 forced-k=4 clusters (see
investigate_k4_insulin_dependent.py) an insulin_dependent participant falls
into? One-vs-rest AUPRC per cluster, ElasticNet and XGBoost, against each
cluster's own prevalence baseline.

Only the curated VIF-pruned CGM feature set (20 features, already excludes
gmi_pct) is used here, not the full 66-feature catalog -- with only 245
participants split 4 ways (cluster 2 alone is 34), the full feature set risks
overfitting badly, and cluster 2 in particular is a small, thin positive
class (see conversation: predicting it from CGM is already a near-tautological,
underpowered exercise given cluster 2 is essentially "the high-HbA1c cluster").
This script exists to make that concrete with actual numbers, not to
recommend the result as a robust finding.

First recomputes and saves the k=4 fit's cluster_assignments_k4.csv (same
method as clustering_matrix.py, just k=4 forced instead of silhouette-chosen
k=2), since that was only ever produced ad hoc in a scratch script before.

Outputs (in cluster_matrix/insulin_dependent/):
  cluster_assignments_k4.csv
  cgm_predicts_cluster_k4_results.csv
  cgm_predicts_cluster_k4_auprc_forest.png
"""
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

CM_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CM_DIR)
sys.path.insert(0, CM_DIR)
_spec = importlib.util.spec_from_file_location("cm", os.path.join(CM_DIR, "clustering_matrix.py"))
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)
C = cm.C

sys.path.insert(0, os.path.join(ROOT, "classification"))
from ablation_common import MODEL_TYPE_STYLE, bootstrap_ci, fit_elasticnet_classifier, fit_xgb_classifier, load_data

sys.path.insert(0, os.path.join(ROOT, "regression_no_correlated_features"))
from select_cgm_features_vif import CGM_FEATURES_VIF_PRUNED

SLUG = "insulin_dependent"
K_FORCED = 4
FITTERS = {"ElasticNet": fit_elasticnet_classifier, "XGBoost": fit_xgb_classifier}


def fit_k4_clusters():
    """Same method as clustering_matrix.py (outlier removal -> k-NN graph ->
    spectral embedding -> KMeans), k forced to 4 instead of silhouette-chosen."""
    df = C.load_raw()
    df = C.filter_group(df, SLUG)
    df = df.dropna(subset=C.FEATURE_COLS).reset_index(drop=True)

    X = df[C.FEATURE_COLS].copy()
    for c in C.LOG_COLS:
        X[c] = np.log1p(X[c])

    idc = C.id_col(df)
    df, X, outliers_df = cm.flag_and_remove_outliers(df, X, idc)
    print(f"[{SLUG}] n={len(df)} after removing {len(outliers_df)} outlier(s)")

    X_scaled = StandardScaler().fit_transform(X)
    affinity, sigma, n_neighbors, n_components = cm.connected_neighbor_graph(X_scaled)
    emb = cm.spectral_embedding(affinity, K_FORCED)
    raw_labels = KMeans(n_clusters=K_FORCED, n_init=10, random_state=0).fit_predict(emb)

    composite_idx = [C.FEATURE_COLS.index(c) for c in (C.BMI_COL, C.INSULIN_COL, C.CPEPTIDE_COL)]
    composite_score = X_scaled[:, composite_idx].mean(axis=1)
    cluster_order = pd.Series(composite_score).groupby(raw_labels).mean().sort_values().index.tolist()
    relabel_map = {old: new for new, old in enumerate(cluster_order)}
    df["cluster"] = pd.Series(raw_labels).map(relabel_map).values

    out_path = os.path.join(C.out_dir(SLUG), "cluster_assignments_k4.csv")
    df[[idc, "cluster"]].to_csv(out_path, index=False)
    print(f"saved {out_path}")
    return df[[idc, "cluster"]]


def sensitivity_score(y_true, y_score, threshold=0.5):
    y_true, y_pred = np.asarray(y_true), (np.asarray(y_score) >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp / (tp + fn) if (tp + fn) else float("nan")


def run_cluster(cluster_id, train, test, feats):
    X_train, X_test = train[feats], test[feats]
    y_train = (train["cluster"] == cluster_id).astype(int).values
    y_test = (test["cluster"] == cluster_id).astype(int).values
    prevalence = float(y_test.mean())
    print(f"[cluster {cluster_id}] n_train={len(train)} ({y_train.sum()} pos)  "
          f"n_test={len(test)} ({y_test.sum()} pos, prevalence={prevalence:.3f})")

    rows = []
    for model_type, fitter in FITTERS.items():
        model = fitter(X_train, y_train)
        proba_test = model.predict_proba(X_test)[:, 1]
        auprc = average_precision_score(y_test, proba_test)
        auprc_lo, auprc_hi = bootstrap_ci(y_test, proba_test, average_precision_score)
        sens = sensitivity_score(y_test, proba_test)
        sens_lo, sens_hi = bootstrap_ci(y_test, proba_test, sensitivity_score)
        rows.append({
            "cluster": cluster_id, "model_type": model_type, "n_test": len(test),
            "n_test_pos": int(y_test.sum()), "prevalence": prevalence,
            "auprc": auprc, "auprc_lo": auprc_lo, "auprc_hi": auprc_hi,
            "sensitivity": sens, "sensitivity_lo": sens_lo, "sensitivity_hi": sens_hi,
        })
        print(f"  {model_type:11s} AUPRC={auprc:.3f} [{auprc_lo:.3f},{auprc_hi:.3f}]  "
              f"sens={sens:.3f} [{sens_lo:.3f},{sens_hi:.3f}]")
    return rows


def make_forest_plot(results_df, out_path):
    clusters = sorted(results_df["cluster"].unique())
    model_types = list(MODEL_TYPE_STYLE.keys())
    dodge = {model_types[0]: 0.12, model_types[1]: -0.12}
    n_clusters = len(clusters)
    y_base = np.arange(n_clusters)[::-1]

    fig, ax = plt.subplots(figsize=(8.5, 1.1 * n_clusters + 1.8), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    prev_by_cluster = results_df.drop_duplicates("cluster").set_index("cluster")["prevalence"]
    for ci, cl in enumerate(clusters):
        y = y_base[ci]
        ax.plot(
            [prev_by_cluster[cl]] * 2, [y - 0.32, y + 0.32],
            color="#999999", linewidth=1.6, linestyle="--", zorder=2,
            label="prevalence (no-skill baseline)" if ci == 0 else None,
        )

    for mt in model_types:
        sub = results_df[results_df["model_type"] == mt].set_index("cluster").loc[clusters]
        y = y_base + dodge[mt]
        x = sub["auprc"].values
        lo = x - sub["auprc_lo"].values
        hi = sub["auprc_hi"].values - x
        ax.errorbar(
            x, y, xerr=[lo, hi], fmt=MODEL_TYPE_STYLE[mt]["marker"],
            color=MODEL_TYPE_STYLE[mt]["color"], ecolor=MODEL_TYPE_STYLE[mt]["color"],
            elinewidth=1.4, capsize=3, markersize=9, markeredgecolor="#0b0b0b",
            markeredgewidth=0.6, label=mt, zorder=3, clip_on=False,
        )

    n_by_cluster = results_df.drop_duplicates("cluster").set_index("cluster")["n_test"]
    npos_by_cluster = results_df.drop_duplicates("cluster").set_index("cluster")["n_test_pos"]
    labels = [f"Cluster {cl} (n_test={n_by_cluster[cl]}, {npos_by_cluster[cl]} pos)" for cl in clusters]
    ax.set_yticks(y_base)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUPRC (one-vs-rest)", fontsize=11, color="#0b0b0b")
    ax.set_title(
        "insulin_dependent: CGM-only prediction of k=4 cluster membership\n"
        "curated CGM feature set, AUPRC one-vs-rest, 95% CI",
        fontsize=11.5, color="#0b0b0b", pad=12,
    )
    ax.xaxis.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.58, 1.0),
               ncol=len(handles), frameon=False, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    assign = fit_k4_clusters()
    df = load_data()
    idc = C.id_col(df) if hasattr(C, "id_col") else "participant_id"
    merged = df.merge(assign, on=idc, how="inner")
    print(f"[{SLUG}] n={len(merged)} after joining CGM-QC-filtered data with k=4 cluster assignments")

    train = merged[merged["split"] == "train"]
    test = merged[merged["split"] == "test"]
    print(f"n_train={len(train)}, n_test={len(test)}")
    print("train cluster counts:", train["cluster"].value_counts().sort_index().to_dict())
    print("test cluster counts:", test["cluster"].value_counts().sort_index().to_dict())

    all_rows = []
    for cluster_id in range(K_FORCED):
        all_rows.extend(run_cluster(cluster_id, train, test, CGM_FEATURES_VIF_PRUNED))
    results_df = pd.DataFrame(all_rows)

    out_dir = C.out_dir(SLUG)
    csv_path = os.path.join(out_dir, "cgm_predicts_cluster_k4_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"saved {csv_path}")

    make_forest_plot(results_df, os.path.join(out_dir, "cgm_predicts_cluster_k4_auprc_forest.png"))


if __name__ == "__main__":
    main()
