"""
Shared helper for plotting pairwise-ROC curves: for every pair of study
groups, treat a single lab value as a univariate classifier score and plot
its ROC curve. One curve per pair, each pair with its own color, all on the
same axes.
"""
import itertools

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

STUDY_GROUPS = [
    "healthy",
    "pre_diabetes_lifestyle_controlled",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled",
    "insulin_dependent",
]
GROUP_LABELS = {
    "healthy": "Healthy",
    "pre_diabetes_lifestyle_controlled": "Pre-diabetic",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled": "Oral/non-insulin med",
    "insulin_dependent": "Insulin-dependent",
}

PAIRS = list(itertools.combinations(STUDY_GROUPS, 2))
PAIR_COLORS = dict(zip(PAIRS, plt.get_cmap("tab10").colors))


def make_pairwise_roc_plot(df, score_col, title, out_path):
    """df must have a 'study_group' column. For each pair (neg, pos) in
    PAIRS, the later group in STUDY_GROUPS order is treated as the positive
    class."""
    fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for neg, pos in PAIRS:
        sub = df[df["study_group"].isin([neg, pos])]
        y = (sub["study_group"] == pos).astype(int)
        score = sub[score_col]
        if y.nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(y, score)
        auc = roc_auc_score(y, score)
        ax.plot(
            fpr, tpr, color=PAIR_COLORS[(neg, pos)], linewidth=2,
            label=f"{GROUP_LABELS[neg]} vs {GROUP_LABELS[pos]} (AUC={auc:.2f})",
        )
        print(f"{GROUP_LABELS[neg]:22s} vs {GROUP_LABELS[pos]:22s} AUC={auc:.3f}  n={len(sub)}")

    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1, zorder=0)
    ax.set_xlabel("False positive rate", fontsize=11)
    ax.set_ylabel("True positive rate", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
