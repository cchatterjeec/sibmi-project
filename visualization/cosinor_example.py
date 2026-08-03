"""
Educational plot: the 24h cosinor fit (Cornelissen 2014) -- mesor, amplitude,
acrophase -- overlaid on the raw glucose-vs-hour-of-day scatter it was fit
to. Reuses cgm_ml_features.cosinor() directly so the fit matches the
pipeline exactly.

Output: cosinor_example.png
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from cgm_ml_features import load_all, load_series, cosinor

PARTICIPANT_ID = "7462"


def main():
    cgm = load_all(os.path.join(ROOT, "ai_readi", "preprocessed", "cgm.parquet"))
    sub = cgm[cgm["participant_id"] == PARTICIPANT_ID]
    vals, times = load_series(sub)
    hod = np.array([t.hour + t.minute / 60.0 for t in times])

    mesor, amp, acro = cosinor(vals, times)

    hh = np.linspace(0, 24, 400)
    w = 2 * np.pi / 24.0
    # matches cgm_ml_features.cosinor(): acro is the peak hour, i.e.
    # curve = mesor + amp*cos(w*(t - acro))
    fit = mesor + amp * np.cos(w * (hh - acro))

    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.scatter(hod, vals, s=4, color="#9aa5b1", alpha=0.35, zorder=1, label="Raw readings (all days, by hour-of-day)")
    ax.plot(hh, fit, color="#1d1d1d", linewidth=2.4, zorder=3, label="Fitted 24h cosinor curve")

    ax.axhline(mesor, color="#4c72b0", linewidth=1.6, linestyle="--", zorder=2)
    ax.text(0.3, mesor + 4, f"mesor = {mesor:.0f} mg/dL", color="#4c72b0", fontsize=10)

    peak_val = mesor + amp
    ax.annotate(
        "", xy=(acro, peak_val), xytext=(acro, mesor),
        arrowprops=dict(arrowstyle="<->", color="#ee8434", linewidth=1.6), zorder=4,
    )
    ax.text(acro + 0.4, (mesor + peak_val) / 2, f"amplitude\n= {amp:.0f} mg/dL", color="#ee8434", fontsize=10)

    ax.axvline(acro, color="#d62728", linewidth=1.2, linestyle=":", zorder=2)
    ax.text(acro + 0.3, ax.get_ylim()[0] if False else vals.min() - 15,
            f"acrophase\n= {acro:.1f}h", color="#d62728", fontsize=10, ha="left")

    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlabel("Local hour of day", fontsize=11)
    ax.set_ylabel("Sensor glucose (mg/dL)", fontsize=11)
    ax.set_title(
        f"Participant {PARTICIPANT_ID} -- 24h cosinor fit\n"
        f"mesor (rhythm-adjusted mean) = {mesor:.1f}   |   amplitude = {amp:.1f}   |   acrophase (peak hour) = {acro:.1f}h",
        fontsize=12.5, pad=14,
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out_path = os.path.join(HERE, "cosinor_example.png")
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[OUT] wrote {out_path}")
    print(f"mesor={mesor:.1f} amplitude={amp:.1f} acrophase={acro:.1f}")


if __name__ == "__main__":
    main()
