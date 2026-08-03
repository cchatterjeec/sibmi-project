"""
Educational plot: the "dawn effect" (dawn_rise = mean(06-09h) - mean(00-06h))
illustrated as an hour-of-day averaged glucose profile -- every reading is
binned by its local hour-of-day and averaged across all of a participant's
days, so the systematic morning rise is visible without single-day noise.

Output: dawn_effect_example.png
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


def main():
    cgm = load_all(os.path.join(ROOT, "ai_readi", "preprocessed", "cgm.parquet"))
    sub = cgm[cgm["participant_id"] == PARTICIPANT_ID]
    vals, times = load_series(sub)

    hod = np.array([t.hour + t.minute / 60.0 for t in times])
    bin_idx = np.floor(hod).astype(int)
    bin_mean = np.array([vals[bin_idx == h].mean() for h in range(24)])
    bin_sd = np.array([vals[bin_idx == h].std() for h in range(24)])

    overnight = vals[(hod >= 0) & (hod < 6)]
    dawn = vals[(hod >= 6) & (hod < 9)]
    overnight_mean, dawn_mean = overnight.mean(), dawn.mean()
    dawn_rise = dawn_mean - overnight_mean

    fig, ax = plt.subplots(figsize=(13, 6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.axvspan(0, 6, color="#4c72b0", alpha=0.15, label="Overnight window (00-06h)")
    ax.axvspan(6, 9, color="#ee8434", alpha=0.20, label="Dawn window (06-09h)")

    hours = np.arange(24)
    ax.fill_between(hours, bin_mean - bin_sd, bin_mean + bin_sd, color="#1d1d1d", alpha=0.08)
    ax.plot(hours, bin_mean, color="#1d1d1d", linewidth=2, marker="o", markersize=4, zorder=3)

    ax.hlines(overnight_mean, 0, 6, color="#4c72b0", linewidth=3, zorder=4)
    ax.hlines(dawn_mean, 6, 9, color="#ee8434", linewidth=3, zorder=4)
    ax.annotate(
        "", xy=(7.5, dawn_mean), xytext=(7.5, overnight_mean),
        arrowprops=dict(arrowstyle="<->", color="#333333", linewidth=1.4),
    )
    ax.text(7.9, (overnight_mean + dawn_mean) / 2, f"dawn_rise\n= {dawn_rise:+.1f} mg/dL",
            fontsize=10, color="#333333", va="center")

    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Local hour of day", fontsize=11)
    ax.set_ylabel("Sensor glucose (mg/dL), averaged across all days", fontsize=11)
    ax.set_title(
        f"Participant {PARTICIPANT_ID} -- dawn effect: hour-of-day averaged glucose profile\n"
        f"Overnight mean (00-06h) = {overnight_mean:.1f} mg/dL   |   "
        f"Dawn mean (06-09h) = {dawn_mean:.1f} mg/dL   |   dawn_rise = {dawn_rise:.1f} mg/dL",
        fontsize=12.5, pad=14,
    )
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out_path = os.path.join(HERE, "dawn_effect_example.png")
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[OUT] wrote {out_path}")
    print(f"overnight_mean={overnight_mean:.1f} dawn_mean={dawn_mean:.1f} dawn_rise={dawn_rise:.1f}")


if __name__ == "__main__":
    main()
