"""
Educational plot: MAGE (Mean Amplitude of Glycemic Excursions, Service 1970)
illustrated on a short window of raw CGM. Shows the turning points (local
peaks/troughs) the algorithm finds, and which excursions between them are
large enough (> 1 SD of the whole series) to count toward MAGE.

Reuses cgm_ml_features.mage()'s exact turning-point logic so the annotated
value matches the pipeline's computed feature.

Output: mage_example.png
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from cgm_ml_features import load_all, load_series

PARTICIPANT_ID = "7462"
WINDOW_DAYS = (0, 3)  # first 3 days, for readability -- turning points are easy to see


def turning_points(g):
    """Same logic as cgm_ml_features.mage(): local extrema of the series."""
    dd = np.diff(g)
    tp = [0]
    for i in range(1, len(dd)):
        if dd[i - 1] != 0 and dd[i] != 0 and np.sign(dd[i]) != np.sign(dd[i - 1]):
            tp.append(i)
    tp.append(len(g) - 1)
    return np.array(tp)


def main():
    cgm = load_all(os.path.join(ROOT, "ai_readi", "preprocessed", "cgm.parquet"))
    sub = cgm[cgm["participant_id"] == PARTICIPANT_ID]
    vals, times = load_series(sub)
    sd_full = float(np.std(vals, ddof=1))

    t0 = times[0]
    days_all = np.array([(t - t0).total_seconds() / 86400.0 for t in times])
    mask = (days_all >= WINDOW_DAYS[0]) & (days_all <= WINDOW_DAYS[1])
    g = vals[mask]
    days = days_all[mask]

    tp = turning_points(g)
    tp_days, tp_vals = days[tp], g[tp]
    amps = np.abs(np.diff(tp_vals))
    counted = amps > sd_full

    fig, ax = plt.subplots(figsize=(15, 6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(days, g, color="#1d1d1d", linewidth=1.1, zorder=2)
    ax.scatter(tp_days, tp_vals, color="#555555", s=18, zorder=4, label="Turning point (local peak/trough)")

    for i in range(len(tp) - 1):
        x0, x1 = tp_days[i], tp_days[i + 1]
        y0, y1 = tp_vals[i], tp_vals[i + 1]
        color = "#d62728" if counted[i] else "#c7c7c7"
        lw = 2.2 if counted[i] else 1.2
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, alpha=0.85, zorder=3)
        if counted[i]:
            xm, ym = (x0 + x1) / 2, (y0 + y1) / 2 + 8
            ax.annotate(f"{amps[i]:.0f}", (xm, ym), fontsize=8, color="#d62728", ha="center")

    ax.axhline(g.mean(), color="#888888", linewidth=0.6, linestyle=":")
    mage_val = float(np.mean(amps[counted])) if counted.any() else float("nan")

    handles = [
        plt.Line2D([0], [0], color="#d62728", linewidth=2.2, label=f"Excursion > 1 SD ({sd_full:.0f} mg/dL) -- counted in MAGE"),
        plt.Line2D([0], [0], color="#c7c7c7", linewidth=1.2, label="Excursion <= 1 SD -- not counted"),
        plt.Line2D([0], [0], marker="o", color="#555555", linewidth=0, label="Turning point"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=9.5)

    ax.set_xlabel("Days of CGM wear", fontsize=11)
    ax.set_ylabel("Sensor glucose (mg/dL)", fontsize=11)
    ax.set_title(
        f"Participant {PARTICIPANT_ID} -- MAGE illustrated on days {WINDOW_DAYS[0]}-{WINDOW_DAYS[1]}\n"
        f"Whole-series SD = {sd_full:.1f} mg/dL (the >1 SD threshold)   |   "
        f"MAGE = mean of the counted (red) excursion sizes = {mage_val:.1f} mg/dL",
        fontsize=12.5, pad=14,
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out_path = os.path.join(HERE, "mage_example.png")
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[OUT] wrote {out_path}")
    print(f"windowed MAGE illustration value: {mage_val:.1f} (full-series pipeline mage may differ slightly -- this window is a 3-day subset)")


if __name__ == "__main__":
    main()
