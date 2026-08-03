"""
Stage 2, stratified: predict membership in the within-group, auto-identified
kidney-complication cluster (meta.json: target_cluster_kmeans) directly from
CGM features alone -- no demographics, no labs. Same non-circularity argument
as clusters_9_vars/predict_target_cluster.py: none of the 9 clustering
variables are glucose measurements.

AUPRC (average precision) is the PRIMARY metric here, not AUROC: the target
class is rare (the whole-cohort version was 45/2155 ~=2%, and stratifying
narrows the cohort further), and AUROC is insensitive to class imbalance in
a way that can look deceptively strong when the positive class is this thin.
AUPRC's baseline is the actual prevalence (plotted as the "chance" line,
not 0.5), which is the honest reference point under imbalance. AUROC is
still reported as a secondary/supplementary metric. Model selection (PLS-DA
component count, ElasticNet/XGBoost hyperparameters) is also tuned on
average_precision throughout, for consistency.

If a group's target cluster has too few members to fit anything meaningful
(train positives < 8 or test positives < 3), modeling is skipped and a
minimal results file records why, rather than reporting an unstable AUPRC/
AUROC off a handful of positives.

Outputs (in this directory):
  target_cluster_results.csv
  target_cluster_auprc_forest.png (primary)
  target_cluster_auroc_forest.png (secondary)
  target_cluster_pr_curve.png (primary)
  target_cluster_roc_curve.png (secondary)
  target_cluster_confusion_matrices.png
  target_cluster_predicted_probability_3d.png
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (average_precision_score, confusion_matrix,
                              precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

sys.path.insert(0, os.path.join(C.ROOT, "regression_no_correlated_features"))
from select_cgm_features_vif import CGM_FEATURES_VIF_PRUNED

MIN_TRAIN_POS = 8
MIN_TEST_POS = 3
VIEW_ELEV, VIEW_AZIM = 20, 45
MODEL_STYLE = {
    "ElasticNet": {"color": "#2a78d6", "marker": "o"},
    "XGBoost": {"color": "#eb6834", "marker": "s"},
    "PLS-DA": {"color": "#2ca858", "marker": "^"},
}


def bootstrap_ci(y_true, y_score, metric_fn, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if yt.min() == yt.max():
            continue
        try:
            stats.append(metric_fn(yt, ys))
        except (ValueError, ZeroDivisionError):
            continue
    if len(stats) < 20:
        return np.nan, np.nan
    return np.percentile(stats, [2.5, 97.5])


def fit_elasticnet(X_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegressionCV(
            Cs=10, cv=5, penalty="elasticnet", solver="saga",
            l1_ratios=[0.1, 0.5, 0.7, 0.9, 1.0], max_iter=5000,
            scoring="average_precision", n_jobs=-1, random_state=0,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def fit_xgb(X_train, y_train):
    base = XGBClassifier(eval_metric="aucpr", random_state=0, n_jobs=1)
    grid = {"max_depth": [2, 3, 4], "n_estimators": [100, 200], "learning_rate": [0.03, 0.1]}
    gs = GridSearchCV(base, grid, scoring="average_precision", cv=5, n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_


class PLSDAClassifier:
    def __init__(self, n_components):
        self.n_components = n_components
        self.model = PLSRegression(n_components=n_components)

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        self.model.fit(self.scaler.transform(X), y)
        return self

    def decision_score(self, X):
        return self.model.predict(self.scaler.transform(X)).ravel()

    def predict(self, X):
        return (self.decision_score(X) >= 0.5).astype(int)


def fit_plsda(X_train, y_train):
    best_score, best_k = -np.inf, 1
    for k in range(1, min(11, X_train.shape[1]) + 1):
        scaler = StandardScaler().fit(X_train)
        Xs = scaler.transform(X_train)
        pls = PLSRegression(n_components=k)
        scores = []
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(y_train))
        folds = np.array_split(idx, 5)
        for i in range(5):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            pls.fit(Xs[train_idx], y_train.values[train_idx])
            pred = pls.predict(Xs[test_idx]).ravel()
            try:
                scores.append(average_precision_score(y_train.values[test_idx], pred))
            except ValueError:
                continue
        mean_score = np.mean(scores) if scores else -np.inf
        if mean_score > best_score:
            best_score, best_k = mean_score, k
    print(f"  PLS-DA: chosen n_components={best_k} (CV AUPRC={best_score:.3f})")
    return PLSDAClassifier(best_k).fit(X_train, y_train)


def write_insufficient_data(out_dir_, slug, reason, **counts):
    pd.DataFrame([{"model_type": "none", "status": "skipped", "reason": reason, **counts}]).to_csv(
        os.path.join(out_dir_, "target_cluster_results.csv"), index=False
    )
    print(f"[{slug}] SKIPPED modeling: {reason} ({counts})")


def main(slug):
    OUT_DIR = C.out_dir(slug)
    meta = C.load_meta(slug)
    target_cluster = meta["target_cluster_kmeans"]

    df = C.load_raw(require_cgm_qc=True)
    df = C.filter_group(df, slug)

    base_assignments = pd.read_csv(os.path.join(OUT_DIR, "clustering_base_assignments.csv"))
    idc = base_assignments.columns[0]
    n_target_total = int((base_assignments["kmeans_cluster"] == target_cluster).sum())
    df = df.merge(base_assignments[[idc, "kmeans_cluster", "pc1", "pc2", "pc3"]], on=idc, how="inner")
    df = df.dropna(subset=CGM_FEATURES_VIF_PRUNED).reset_index(drop=True)
    df["y"] = (df["kmeans_cluster"] == target_cluster).astype(int)

    print(f"[{slug}] cohort: n={len(df)} (target=1: n={df['y'].sum()} of {n_target_total} total "
          f"kidney-cluster-{target_cluster} members, target=0: n={(df['y']==0).sum()})")

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    y_train, y_test = train["y"], test["y"].values
    print(f"[{slug}] train n={len(train)} (target=1: {y_train.sum()})  test n={len(test)} (target=1: {y_test.sum()})")

    if y_train.sum() < MIN_TRAIN_POS or y_test.sum() < MIN_TEST_POS:
        write_insufficient_data(
            OUT_DIR, slug,
            reason=f"fewer than {MIN_TRAIN_POS} train / {MIN_TEST_POS} test positives -- "
                    "AUPRC/AUROC would be unstable off this few cases",
            train_n=len(train), train_pos=int(y_train.sum()), test_n=len(test), test_pos=int(y_test.sum()),
        )
        return

    X_train, X_test = train[CGM_FEATURES_VIF_PRUNED], test[CGM_FEATURES_VIF_PRUNED]
    prevalence = float(y_test.mean())

    fitters = {"ElasticNet": fit_elasticnet, "XGBoost": fit_xgb, "PLS-DA": fit_plsda}
    rows = []
    scores_by_model = {}
    models_by_name = {}

    for model_name, fitter in fitters.items():
        print(f"[{slug}] fitting {model_name}...")
        try:
            model = fitter(X_train, y_train)
        except Exception as e:
            print(f"[{slug}] {model_name} failed to fit: {e}")
            continue
        models_by_name[model_name] = model
        proba_test = model.decision_score(X_test) if model_name == "PLS-DA" else model.predict_proba(X_test)[:, 1]
        scores_by_model[model_name] = proba_test

        auprc = average_precision_score(y_test, proba_test)
        auprc_lo, auprc_hi = bootstrap_ci(y_test, proba_test, average_precision_score)
        auroc = roc_auc_score(y_test, proba_test)
        auroc_lo, auroc_hi = bootstrap_ci(y_test, proba_test, roc_auc_score)
        rows.append({
            "model_type": model_name,
            "auprc": auprc, "auprc_lo": auprc_lo, "auprc_hi": auprc_hi,
            "auroc": auroc, "auroc_lo": auroc_lo, "auroc_hi": auroc_hi,
            "prevalence": prevalence,
        })
        print(f"[{slug}] {model_name}: AUPRC={auprc:.3f} [{auprc_lo:.3f},{auprc_hi:.3f}]  "
              f"(baseline={prevalence:.3f})  AUROC={auroc:.3f} [{auroc_lo:.3f},{auroc_hi:.3f}]")

    if not rows:
        write_insufficient_data(OUT_DIR, slug, reason="all model fits failed", train_n=len(train), test_n=len(test))
        return

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(OUT_DIR, "target_cluster_results.csv"), index=False)

    def forest_plot(metric, lo_col, hi_col, baseline, baseline_label, fname, title):
        fig, ax = plt.subplots(figsize=(7.5, 4), facecolor="#fcfcfb")
        ax.set_facecolor("#fcfcfb")
        ax.axvline(baseline, color="#999999", linewidth=1.0, linestyle="--", zorder=1)
        ax.text(baseline + 0.01, len(rows) - 0.4, baseline_label, color="#999999", fontsize=8.5, va="center")
        y_pos = np.arange(len(rows))[::-1]
        for i, row in enumerate(rows):
            style = MODEL_STYLE[row["model_type"]]
            lo, hi = row[lo_col], row[hi_col]
            xerr = [[0], [0]] if np.isnan(lo) else [[row[metric] - lo], [hi - row[metric]]]
            ax.errorbar(row[metric], y_pos[i], xerr=xerr,
                        fmt=style["marker"], color=style["color"], ecolor=style["color"], elinewidth=1.4, capsize=3,
                        markersize=10, markeredgecolor="#0b0b0b", markeredgewidth=0.6, zorder=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([row["model_type"] for row in rows], fontsize=11)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(metric.upper(), fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.xaxis.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, fname), dpi=200, facecolor=fig.get_facecolor())
        plt.close(fig)

    forest_plot("auprc", "auprc_lo", "auprc_hi", prevalence, f"baseline (prevalence={prevalence:.3f})",
                "target_cluster_auprc_forest.png",
                f"{slug}: predicting kidney-complication cluster from CGM alone\n(AUPRC, primary metric under class imbalance)")
    forest_plot("auroc", "auroc_lo", "auroc_hi", 0.5, "chance",
                "target_cluster_auroc_forest.png",
                f"{slug}: predicting kidney-complication cluster from CGM alone\n(AUROC, secondary metric)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#fcfcfb")
    for ax, model_name in zip(axes, [r["model_type"] for r in rows]):
        proba = scores_by_model[model_name]
        pred_class = (proba >= 0.5).astype(int)
        cm = confusion_matrix(y_test, pred_class, labels=[0, 1])
        ax.set_facecolor("#fcfcfb")
        ax.imshow(cm, cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16,
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Not kidney", "Kidney"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Not kidney", "Kidney"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(model_name, fontsize=12)
    fig.suptitle(f"{slug}: confusion matrices (threshold 0.5), predicting kidney cluster from CGM", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "target_cluster_confusion_matrices.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.axhline(prevalence, color="#999999", linewidth=1.0, linestyle="--", zorder=1, label=f"Baseline (prevalence={prevalence:.3f})")
    for row in rows:
        model_name = row["model_type"]
        precision, recall, _ = precision_recall_curve(y_test, scores_by_model[model_name])
        style = MODEL_STYLE[model_name]
        ax.plot(recall, precision, color=style["color"], linewidth=2.0, zorder=3, label=f"{model_name} (AUPRC={row['auprc']:.3f})")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(f"{slug}: precision-recall, predicting kidney cluster from CGM alone", fontsize=12)
    ax.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "target_cluster_pr_curve.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot([0, 1], [0, 1], color="#999999", linewidth=1.0, linestyle="--", zorder=1, label="Chance")
    for row in rows:
        model_name = row["model_type"]
        fpr, tpr, _ = roc_curve(y_test, scores_by_model[model_name])
        style = MODEL_STYLE[model_name]
        ax.plot(fpr, tpr, color=style["color"], linewidth=2.0, zorder=3, label=f"{model_name} (AUROC={row['auroc']:.3f})")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"{slug}: ROC, predicting kidney cluster from CGM alone", fontsize=12)
    ax.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "target_cluster_roc_curve.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)

    best_model_name = results_df.loc[results_df["auprc"].idxmax(), "model_type"]
    best_model = models_by_name[best_model_name]
    X_all = df[CGM_FEATURES_VIF_PRUNED]
    proba_all = best_model.decision_score(X_all) if best_model_name == "PLS-DA" else best_model.predict_proba(X_all)[:, 1]
    df["predicted_proba"] = proba_all

    fig = plt.figure(figsize=(9, 8), facecolor="#fcfcfb")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#fcfcfb")
    sc = ax.scatter(df["pc1"], df["pc2"], df["pc3"], c=df["predicted_proba"], cmap="viridis", s=10, alpha=0.6, edgecolor="none")
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label="Predicted probability of kidney cluster")
    ax.set_xlabel("PC1", fontsize=10); ax.set_ylabel("PC2", fontsize=10); ax.set_zlabel("PC3", fontsize=10)
    ax.set_title(f"{slug}: predicted probability of kidney cluster\n(best model by AUPRC: {best_model_name})", fontsize=12)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "target_cluster_predicted_probability_3d.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"[{slug}] best model (by AUPRC): {best_model_name}")
    print(f"[{slug}] saved target_cluster_results.csv and associated plots")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--group", required=True, choices=list(C.GROUPS))
    args = p.parse_args()
    main(args.group)
