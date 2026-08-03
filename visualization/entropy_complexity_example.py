"""
Educational plot: sample_entropy contrasted between two participants with
very different values, to show that entropy measures REGULARITY/
predictability moment-to-moment, not the AMOUNT of variability (that's
cv_pct/sd_glucose's job). Deliberately chosen so the point is visible: the
low-entropy participant actually has *larger* swings (higher cv_pct) than
the high-entropy one -- just more repetitive/patterned swings, vs smaller
but more erratic ones.

Output: entropy_complexity_example.png
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from cgm_ml_features import load_all, load_series, extract

HIGH_ENTROPY_ID = "4026"
LOW_ENTROPY_ID = "4117"
WINDOW_DAYS = (0, 2.5)


def get_window(pid, cgm):
    sub = cgm[cgm["participant_id"] == pid]
    vals, times = load_series(sub)
    feat = extract(vals, times)
    t0 = times[0]
    days_all = np.array([(t - t0).total_seconds() / 86400.0 for t in times])
    mask = (days_all >= WINDOW_DAYS[0]) & (days_all <= WINDOW_DAYS[1])
    return days_all[mask], vals[mask], feat


def main():
    cgm = load_all(os.path.join(ROOT, "ai_readi", "preprocessed", "cgm.parquet"))
    days_hi, g_hi, feat_hi = get_window(HIGH_ENTROPY_ID, cgm)
    days_lo, g_lo, feat_lo = get_window(LOW_ENTROPY_ID, cgm)

    y_lo, y_hi = min(g_hi.min(), g_lo.min()) - 10, max(g_hi.max(), g_lo.max()) + 10

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), facecolor="#fcfcfb", sharex=True, sharey=True)

    for ax, days, g, feat, pid, color, label in [
        (axes[0], days_hi, g_hi, feat_hi, HIGH_ENTROPY_ID, "#2a78d6", "HIGH sample_entropy (more erratic/less predictable)"),
        (axes[1], days_lo, g_lo, feat_lo, LOW_ENTROPY_ID, "#eb6834", "LOW sample_entropy (more regular/predictable)"),
    ]:
        ax.set_facecolor("#fcfcfb")
        ax.plot(days, g, color=color, linewidth=1.3)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(
            f"Participant {pid} -- {label}\n"
            f"sample_entropy = {feat['sample_entropy']:.3f}   |   "
            f"cv_pct = {feat['cv_pct']:.1f}%   |   mean_glucose = {feat['mean_glucose']:.0f} mg/dL",
            fontsize=11.5, pad=10,
        )
        ax.set_ylabel("Glucose (mg/dL)", fontsize=10)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[1].set_xlabel("Days of CGM wear", fontsize=11)
    fig.suptitle(
        "Entropy vs. variability are different things:\n"
        "the LOW-entropy trace below actually swings MORE (higher CV) -- its ups and downs are just more repetitive/patterned",
        fontsize=12, y=1.02,
    )

    fig.tight_layout()
    out_path = os.path.join(HERE, "entropy_complexity_example.png")
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[OUT] wrote {out_path}")
    print(f"HIGH entropy participant {HIGH_ENTROPY_ID}: entropy={feat_hi['sample_entropy']:.3f} cv={feat_hi['cv_pct']:.1f}")
    print(f"LOW entropy participant {LOW_ENTROPY_ID}: entropy={feat_lo['sample_entropy']:.3f} cv={feat_lo['cv_pct']:.1f}")


if __name__ == "__main__":
    main()
