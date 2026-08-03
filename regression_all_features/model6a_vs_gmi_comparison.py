"""
Head-to-head comparison: Model 6a ElasticNet (demographics + BMI +
wearables + full CGM feature catalog minus gmi_pct, R2=0.661) vs. gmi_pct
alone (the algebraic "estimated A1c" formula, GMI = 3.31 + 0.02392 *
mean_glucose, Bergenstal 2018) as a direct one-line "prediction" of HbA1c.

Both are doing conceptually the same thing -- estimating lab HbA1c from
CGM data -- so this compares a multivariate model fit on 66 CGM features
(+ demographics/BMI/wearables) against a single fixed formula using only
mean_glucose, on the exact same test set.

gmi_pct requires no fitting (it's a fixed formula, not a model), so its
R2/Spearman rho are computed directly against actual HbA1c with no
train/test split needed -- it's evaluated on the test set only for a fair
apples-to-apples comparison with the ElasticNet fit.

A third panel compares the model's predictions directly against gmi_pct
(not against actual HbA1c) -- this is a redundancy check, not an accuracy
comparison: it asks whether Model 6a's edge over gmi_pct (R2 0.661 vs
0.467) reflects genuinely new information, or whether the model is mostly
just recovering/recalibrating the same mean_glucose-driven signal gmi_pct
already captures (which would show up as a very high predicted-vs-gmi_pct
correlation, e.g. >0.9).

Output: model6a_vs_gmi_comparison.png
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, spearmanr
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ablation_common import (
    CGM_ML_FEATURES,
    HBA1C_COL,
    fit_elasticnet_regressor,
    get_ablation_stages,
    load_data,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CGM_FEATURES_NO_GMI = [c for c in CGM_ML_FEATURES if c != "gmi_pct"]

df = load_data()
stages = get_ablation_stages(
    df, exclude_cpeptide=True, include_no_labs_variants=True, target_col=HBA1C_COL,
    cgm_features=CGM_FEATURES_NO_GMI,
)
stage_name, feats = [s for s in stages if s[0].startswith("Model 6a")][0]

train = df[df["split"] == "train"]
test = df[df["split"] == "test"]
X_train, X_test = train[feats], test[feats]
y_train, y_test = train[HBA1C_COL], test[HBA1C_COL]

model = fit_elasticnet_regressor(X_train, y_train)
pred_model = model.predict(X_test)
pred_gmi = test["gmi_pct"].values

panels = [
    ("ElasticNet, Model 6a\n(66 CGM features + demographics/BMI/wearables)", pred_model, "#2a78d6"),
    ("gmi_pct alone\n(fixed formula: 3.31 + 0.02392 x mean_glucose)", pred_gmi, "#eb6834"),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6.3), facecolor="#fcfcfb")
lims = [
    min(y_test.min(), pred_model.min(), pred_gmi.min()) - 0.3,
    max(y_test.max(), pred_model.max(), pred_gmi.max()) + 0.3,
]

for ax, (title, pred, color) in zip(axes[:2], panels):
    r2 = r2_score(y_test, pred)
    rho, pval = spearmanr(y_test, pred)
    slope, intercept, _, _, _ = linregress(y_test, pred)
    print(f"{title.splitlines()[0]}: R2={r2:.3f}  Spearman rho={rho:.3f} (p={pval:.2e})  slope={slope:.3f}")

    ax.set_facecolor("#fcfcfb")
    ax.plot(lims, lims, color="#999999", linewidth=1.0, linestyle="--", zorder=1, label="Perfect prediction")
    fit_x = np.array(lims)
    ax.plot(fit_x, slope * fit_x + intercept, color=color, linewidth=2.0, zorder=2, label=f"Best fit (slope={slope:.2f})")
    ax.scatter(y_test, pred, s=14, alpha=0.5, color=color, edgecolor="none", zorder=3)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("Actual HbA1c (%)", fontsize=10.5)
    ax.set_ylabel("Predicted / estimated HbA1c (%)", fontsize=10.5)
    ax.set_title(f"{title}\n$R^2$={r2:.3f}  |  Spearman $\\rho$={rho:.3f} (p={pval:.1e})", fontsize=10.5, pad=10)
    ax.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)

# --- Third panel: model prediction vs. gmi_pct directly (redundancy check) ---
ax = axes[2]
r2_redundancy = r2_score(pred_gmi, pred_model)
rho_redundancy, pval_redundancy = spearmanr(pred_gmi, pred_model)
slope_r, intercept_r, _, _, _ = linregress(pred_gmi, pred_model)
print(f"Model prediction vs. gmi_pct (redundancy check): R2={r2_redundancy:.3f}  "
      f"Spearman rho={rho_redundancy:.3f} (p={pval_redundancy:.2e})  slope={slope_r:.3f}")

lims_r = [min(pred_gmi.min(), pred_model.min()) - 0.3, max(pred_gmi.max(), pred_model.max()) + 0.3]
ax.set_facecolor("#fcfcfb")
ax.plot(lims_r, lims_r, color="#999999", linewidth=1.0, linestyle="--", zorder=1, label="Identical (y=x)")
fit_x = np.array(lims_r)
ax.plot(fit_x, slope_r * fit_x + intercept_r, color="#2ca858", linewidth=2.0, zorder=2, label=f"Best fit (slope={slope_r:.2f})")
ax.scatter(pred_gmi, pred_model, s=14, alpha=0.5, color="#2ca858", edgecolor="none", zorder=3)
ax.set_xlim(lims_r); ax.set_ylim(lims_r)
ax.set_aspect("equal")
ax.set_xlabel("gmi_pct (estimated HbA1c)", fontsize=10.5)
ax.set_ylabel("ElasticNet Model 6a prediction", fontsize=10.5)
ax.set_title(
    f"Model prediction vs. gmi_pct\n(redundancy check, not vs. actual)\n"
    f"$R^2$={r2_redundancy:.3f}  |  Spearman $\\rho$={rho_redundancy:.3f} (p={pval_redundancy:.1e})",
    fontsize=10.5, pad=10,
)
ax.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.legend(loc="upper left", frameon=False, fontsize=8.5)

fig.suptitle("Model 6a (all CGM features, no gmi_pct) vs. gmi_pct alone -- both estimating lab HbA1c", fontsize=13, y=1.04)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUT_DIR, "model6a_vs_gmi_comparison.png"), dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
print("saved model6a_vs_gmi_comparison.png")
