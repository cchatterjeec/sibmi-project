"""
Same TIR/TAR/TBR band illustration as tir_tar_tbr_example.py (Battelino
2019 consensus bands shaded behind the raw trace), but for a participant
from the pre_diabetes_lifestyle_controlled study_group instead of
insulin_dependent -- meant to sit alongside tir_tar_tbr_example_healthy.py
and the original insulin_dependent plot as a 3-way "what does a
normal/borderline/dysregulated CGM trace actually look like" comparison.

Participant chosen to be representative of the prediabetic group's more
dysregulated tail rather than its literal median: median TAR in this group
is only 1.8% (IQR 0.4-5.6%) -- close to the healthy group's -- since most
prediabetic participants are still well-controlled day to day. Participant
4261 (TIR=91.7%, TAR=8.0%, TBR=0.3%) sits near the group's ~90th
percentile of TAR: still mostly in-range (unlike the insulin_dependent
example), but with clearly visible post-meal excursions above 180 mg/dL --
the pattern that's actually characteristic of impaired glucose tolerance,
rather than an outlier. Full ~10-day/2856-reading wear.

Output: tir_tar_tbr_example_prediabetic.png
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PARTICIPANT_ID = "4261"
STUDY_GROUP_LABEL = "prediabetic"

# Battelino 2019 consensus thresholds (mg/dL)
V_LOW, LOW, HIGH, V_HIGH = 54, 70, 180, 250

BAND_COLORS = {
    "very_low": "#a91409",
    "low": "#e63946",
    "target": "#57a773",
    "high": "#f4d35e",
    "very_high": "#ee8434",
}
LINE_COLOR = "#1d1d1d"


def load_participant(pid):
    cgm = pd.read_parquet(
        os.path.join(ROOT, "ai_readi", "preprocessed", "cgm.parquet"),
        columns=["participant_id", "glucose", "start_time_local", "event_type"],
    )
    sub = cgm[(cgm["participant_id"] == pid) & (cgm["event_type"] == "EGV")].copy()
    sub["start_time_local"] = pd.to_datetime(sub["start_time_local"]).dt.tz_localize(None)
    sub = sub.dropna(subset=["glucose"]).sort_values("start_time_local")
    return sub


def main():
    sub = load_participant(PARTICIPANT_ID)
    t0 = sub["start_time_local"].iloc[0]
    days = (sub["start_time_local"] - t0).dt.total_seconds() / 86400.0
    g = sub["glucose"].to_numpy()

    n = len(g)
    tir = 100 * np.mean((g >= LOW) & (g <= HIGH))
    tar = 100 * np.mean(g > HIGH)
    tar_severe = 100 * np.mean(g > V_HIGH)
    tbr = 100 * np.mean(g < LOW)
    tbr_severe = 100 * np.mean(g < V_LOW)

    fig, ax = plt.subplots(figsize=(16, 6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    y_top = max(300, g.max() + 20)
    bands = [
        (0, V_LOW, BAND_COLORS["very_low"], "Very Low (<54 mg/dL)"),
        (V_LOW, LOW, BAND_COLORS["low"], "Low (54–70 mg/dL)"),
        (LOW, HIGH, BAND_COLORS["target"], "Target Range — TIR (70–180 mg/dL)"),
        (HIGH, V_HIGH, BAND_COLORS["high"], "High (180–250 mg/dL)"),
        (V_HIGH, y_top, BAND_COLORS["very_high"], "Very High (>250 mg/dL)"),
    ]
    for lo, hi, color, _ in bands:
        ax.axhspan(lo, hi, color=color, alpha=0.28, zorder=0)

    ax.plot(days, g, color=LINE_COLOR, linewidth=1.1, zorder=3)

    for thresh in (V_LOW, LOW, HIGH, V_HIGH):
        ax.axhline(thresh, color="#555555", linewidth=0.6, linestyle="--", zorder=1)
        ax.text(days.max() + 0.1, thresh, f"{thresh}", va="center", fontsize=8, color="#555555")

    ax.set_xlim(0, days.max())
    ax.set_ylim(0, y_top)
    ax.set_xlabel("Days of CGM wear", fontsize=11)
    ax.set_ylabel("Sensor glucose (mg/dL)", fontsize=11)
    ax.set_title(
        f"Participant {PARTICIPANT_ID} ({STUDY_GROUP_LABEL}) — {n} readings over {days.max():.1f} days\n"
        f"TIR (70–180) = {tir:.1f}%   |   TAR (>180) = {tar:.1f}% (of which >250 = {tar_severe:.1f}%)"
        f"   |   TBR (<70) = {tbr:.1f}% (of which <54 = {tbr_severe:.1f}%)",
        fontsize=12.5, pad=14,
    )

    handles = [plt.Rectangle((0, 0), 1, 1, color=b[2], alpha=0.5) for b in bands]
    labels = [b[3] for b in bands]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=5, frameon=False, fontsize=9.5)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out_path = os.path.join(HERE, "tir_tar_tbr_example_prediabetic.png")
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[OUT] wrote {out_path}")
    print(f"TIR={tir:.1f}% TAR={tar:.1f}% (severe {tar_severe:.1f}%) TBR={tbr:.1f}% (severe {tbr_severe:.1f}%)")


if __name__ == "__main__":
    main()
