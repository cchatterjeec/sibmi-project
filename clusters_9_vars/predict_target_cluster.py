"""
Stage 2: predict membership in the kidney-complication cluster (9-variable
base KMeans cluster 2 -- markedly elevated creatinine and UACR/albuminuria,
plus elevated CRP and triglycerides, n=54) directly from CGM features.

Unlike clusters_analysis/'s beta-cell-exhaustion target (which was partly
tautological, since CGM directly measures the glucose severity that
defines high HbA1c), NONE of the 9 variables that define this cluster are
glucose measurements -- so this is a genuinely non-circular test: does
CGM's glycemic pattern predict a kidney/lipid/inflammation complication
phenotype it has no definitional overlap with?

Same approach as clusters_analysis/predict_beta_cell_cluster.py: no
ablation staging (the target isn't derived from demographics/labs the
way it would need staging to avoid circularity -- it's derived from a
disjoint set of labs entirely, so this is just a direct, clean CGM-only
fit), ElasticNet + XGBoost + PLS-DA on the curated 20-feature CGM set.

AUPRC (average precision) is the PRIMARY metric, not AUROC: the target
class is rare (test set: 13 of 431), and AUROC is insensitive to class
imbalance in a way that can look deceptively strong for a rare positive
class. AUPRC's baseline is the actual test-set prevalence (plotted as the
"chance" line, not 0.5) -- the honest reference point here. AUROC is
still reported as a secondary metric. Model selection (PLS-DA component
count, ElasticNet/XGBoost hyperparameters) is tuned on average_precision
throughout for consistency.

Cohort: QC-filtered (qc_pct_active >= 70%) participants merged with their
base-clustering label -- of the original 54 cluster-2 members, some may
lack usable CGM data (checked at runtime).

Outputs (in this directory):
  target_cluster_results.csv
  target_cluster_auprc_forest.png (primary)
  target_cluster_auroc_forest.png (secondary)
  target_cluster_pr_curve.png (primary)
  target_cluster_roc_curve.png (secondary)
  target_cluster_confusion_matrices.png
  target_cluster_predicted_probability_3d.png
"""
import os
import re
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

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(OUT_DIR)
DATA_PATH = os.path.join(ROOT, "final_df.csv")
CGM_QC_MIN_PCT_ACTIVE = 70.0
TARGET_CLUSTER = 2
VIEW_ELEV, VIEW_AZIM = 20, 45
MODEL_STYLE = {
    "ElasticNet": {"color": "#2a78d6", "marker": "o"},
    "XGBoost": {"color": "#eb6834", "marker": "s"},
    "PLS-DA": {"color": "#2ca858", "marker": "^"},
}

sys.path.insert(0, os.path.join(ROOT, "regression_no_correlated_features"))
from select_cgm_features_vif import CGM_FEATURES_VIF_PRUNED


def sanitize_column_name(col):
    return re.sub(r"[\[\]<]", "", col)


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


df = pd.read_csv(DATA_PATH)
df.columns = [sanitize_column_name(c) for c in df.columns]
before = len(df)
df = df[df["qc_pct_active"] >= CGM_QC_MIN_PCT_ACTIVE]
print(f"[QC filter] dropped {before - len(df)} participants with qc_pct_active < {CGM_QC_MIN_PCT_ACTIVE}% or missing")

base_assignments = pd.read_csv(os.path.join(OUT_DIR, "clustering_base_assignments.csv"))
n_target_total = int((base_assignments["kmeans_cluster"] == TARGET_CLUSTER).sum())
df = df.merge(base_assignments[["participant_id", "kmeans_cluster", "pc1", "pc2", "pc3"]], on="participant_id", how="inner")
df = df.dropna(subset=CGM_FEATURES_VIF_PRUNED).reset_index(drop=True)
df["y"] = (df["kmeans_cluster"] == TARGET_CLUSTER).astype(int)

print(f"cohort: n={len(df)}  (target=1: n={df['y'].sum()} of {n_target_total} total cluster-{TARGET_CLUSTER} members, "
      f"target=0: n={(df['y']==0).sum()})")

train = df[df["split"] == "train"]
test = df[df["split"] == "test"]
X_train, X_test = train[CGM_FEATURES_VIF_PRUNED], test[CGM_FEATURES_VIF_PRUNED]
y_train, y_test = train["y"], test["y"].values
prevalence = float(y_test.mean())
print(f"train n={len(train)} (target=1: {y_train.sum()})  test n={len(test)} (target=1: {y_test.sum()}, prevalence={prevalence:.3f})")

fitters = {"ElasticNet": fit_elasticnet, "XGBoost": fit_xgb, "PLS-DA": fit_plsda}
rows = []
scores_by_model = {}
models_by_name = {}

for model_name, fitter in fitters.items():
    print(f"fitting {model_name}...")
    model = fitter(X_train, y_train)
    models_by_name[model_name] = model
    if model_name == "PLS-DA":
        proba_test = model.decision_score(X_test)
    else:
        proba_test = model.predict_proba(X_test)[:, 1]
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
    print(f"{model_name}: AUPRC={auprc:.3f}  95% CI=[{auprc_lo:.3f}, {auprc_hi:.3f}]  (baseline={prevalence:.3f})  "
          f"AUROC={auroc:.3f}  95% CI=[{auroc_lo:.3f}, {auroc_hi:.3f}]")

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
        ax.errorbar(row[metric], y_pos[i], xerr=[[row[metric] - row[lo_col]], [row[hi_col] - row[metric]]],
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
            "Predicting kidney-complication cluster membership from CGM alone\n(AUPRC, primary metric under class imbalance)")
forest_plot("auroc", "auroc_lo", "auroc_hi", 0.5, "chance",
            "target_cluster_auroc_forest.png",
            "Predicting kidney-complication cluster membership from CGM alone\n(AUROC, secondary metric)")

fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#fcfcfb")
for ax, model_name in zip(axes, fitters.keys()):
    proba = scores_by_model[model_name]
    pred_class = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred_class, labels=[0, 1])
    ax.set_facecolor("#fcfcfb")
    ax.imshow(cm, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16,
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"Not cluster {TARGET_CLUSTER}", f"Cluster {TARGET_CLUSTER}"])
    ax.set_yticks([0, 1]); ax.set_yticklabels([f"Not cluster {TARGET_CLUSTER}", f"Cluster {TARGET_CLUSTER}"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(model_name, fontsize=12)
fig.suptitle("Confusion matrices (threshold 0.5): predicting kidney-complication cluster from CGM", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "target_cluster_confusion_matrices.png"), dpi=200, facecolor=fig.get_facecolor())
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 6.5), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
ax.axhline(prevalence, color="#999999", linewidth=1.0, linestyle="--", zorder=1, label=f"Baseline (prevalence={prevalence:.3f})")
for model_name in fitters.keys():
    precision, recall, _ = precision_recall_curve(y_test, scores_by_model[model_name])
    auprc = results_df.loc[results_df["model_type"] == model_name, "auprc"].iloc[0]
    style = MODEL_STYLE[model_name]
    ax.plot(recall, precision, color=style["color"], linewidth=2.0, zorder=3, label=f"{model_name} (AUPRC={auprc:.3f})")
ax.set_xlabel("Recall", fontsize=11)
ax.set_ylabel("Precision", fontsize=11)
ax.set_title("Precision-recall: predicting kidney-complication cluster from CGM alone", fontsize=12)
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
for model_name in fitters.keys():
    fpr, tpr, _ = roc_curve(y_test, scores_by_model[model_name])
    auroc = results_df.loc[results_df["model_type"] == model_name, "auroc"].iloc[0]
    style = MODEL_STYLE[model_name]
    ax.plot(fpr, tpr, color=style["color"], linewidth=2.0, zorder=3, label=f"{model_name} (AUROC={auroc:.3f})")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC curves: predicting kidney-complication cluster from CGM alone", fontsize=12)
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
fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label=f"Predicted probability of cluster {TARGET_CLUSTER}")
ax.set_xlabel("PC1", fontsize=10); ax.set_ylabel("PC2", fontsize=10); ax.set_zlabel("PC3", fontsize=10)
ax.set_title(f"Predicted probability of kidney-complication cluster\n(best model by AUPRC: {best_model_name}), on the base-clustering 3D PCA map", fontsize=12)
ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "target_cluster_predicted_probability_3d.png"), dpi=200, facecolor=fig.get_facecolor())
plt.close(fig)

print(f"\nbest model (by AUPRC): {best_model_name}")
print("saved target_cluster_results.csv, target_cluster_auprc_forest.png, target_cluster_auroc_forest.png, "
      "target_cluster_pr_curve.png, target_cluster_roc_curve.png, target_cluster_confusion_matrices.png, "
      "target_cluster_predicted_probability_3d.png")
