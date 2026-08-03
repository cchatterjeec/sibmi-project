"""
Educational plot: same Battelino 2019 consensus glucose bands (TIR / TAR /
TBR) as tir_tar_tbr_example.py, but instead of one participant's raw
10-day trace, this shows one "average day" line per study_group --
every reading from every participant in that group is binned by local
hour-of-day and averaged, producing an hour-of-day glucose profile
(0-24h) per group (same averaging approach as dawn_effect_example.py,
just split 4 ways by study_group instead of computed for one participant).

A single participant's raw trace can't be meaningfully averaged against
another's on a "day of wear" x-axis (day 3 of participant A's wear has no
relationship to day 3 of participant B's), so hour-of-day is the x-axis
that actually lets many participants' readings be pooled into one
comparable curve.

Output: tir_tar_tbr_by_study_group.png
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Battelino 2019 consensus thresholds (mg/dL)
V_LOW, LOW, HIGH, V_HIGH = 54, 70, 180, 250

BAND_COLORS = {
    "very_low": "#a91409",
    "low": "#e63946",
    "target": "#57a773",
    "high": "#f4d35e",
    "very_high": "#ee8434",
}
GROUP_COLORS = {
    "healthy": "#1f77b4",
    "pre_diabetes_lifestyle_controlled": "#9467bd",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled": "#17becf",
    "insulin_dependent": "#000000",
}
GROUP_LABELS = {
    "healthy": "Healthy",
    "pre_diabetes_lifestyle_controlled": "Pre-diabetes",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled": "Oral medication",
    "insulin_dependent": "Insulin dependent",
}


def main():
    labels = pd.read_csv(os.path.join(ROOT, "final_df.csv"), usecols=["participant_id", "study_group"])
    labels["participant_id"] = labels["participant_id"].astype(str)

    cgm = pd.read_parquet(
        os.path.join(ROOT, "ai_readi", "preprocessed", "cgm.parquet"),
        columns=["participant_id", "glucose", "start_time_local", "event_type"],
    )
    cgm = cgm[cgm["event_type"] == "EGV"].dropna(subset=["glucose"]).copy()
    cgm["start_time_local"] = pd.to_datetime(cgm["start_time_local"]).dt.tz_localize(None)
    cgm = cgm.merge(labels, on="participant_id", how="inner")

    hod = cgm["start_time_local"].dt.hour + cgm["start_time_local"].dt.minute / 60.0
    cgm["hour_bin"] = np.floor(hod).astype(int)

    fig, ax = plt.subplots(figsize=(16, 6.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    y_top = 300
    bands = [
        (0, V_LOW, BAND_COLORS["very_low"], "Very Low (<54 mg/dL)"),
        (V_LOW, LOW, BAND_COLORS["low"], "Low (54–70 mg/dL)"),
        (LOW, HIGH, BAND_COLORS["target"], "Target Range — TIR (70–180 mg/dL)"),
        (HIGH, V_HIGH, BAND_COLORS["high"], "High (180–250 mg/dL)"),
        (V_HIGH, y_top, BAND_COLORS["very_high"], "Very High (>250 mg/dL)"),
    ]
    for lo, hi, color, _ in bands:
        ax.axhspan(lo, hi, color=color, alpha=0.28, zorder=0)

    print(f"{'study_group':<70s} {'n_participants':>14s} {'n_readings':>12s} {'TIR%':>6s} {'TAR%':>6s} {'TBR%':>6s}")
    group_lines = []
    for group, color in GROUP_COLORS.items():
        sub = cgm[cgm["study_group"] == group]
        g = sub["glucose"].to_numpy()
        n_participants = sub["participant_id"].nunique()
        tir = 100 * np.mean((g >= LOW) & (g <= HIGH))
        tar = 100 * np.mean(g > HIGH)
        tbr = 100 * np.mean(g < LOW)
        print(f"{group:<70s} {n_participants:>14d} {len(sub):>12d} {tir:>6.1f} {tar:>6.1f} {tbr:>6.1f}")

        bin_mean = sub.groupby("hour_bin")["glucose"].mean().reindex(range(24))
        line, = ax.plot(
            bin_mean.index, bin_mean.values, color=color, linewidth=2.4, marker="o", markersize=4.5, zorder=3,
            label=f"{GROUP_LABELS[group]} (n={n_participants}, TIR={tir:.0f}%)",
        )
        group_lines.append(line)

    ax.set_xlim(0, 23)
    ax.set_ylim(0, y_top)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Local hour of day", fontsize=11)
    ax.set_ylabel("Sensor glucose (mg/dL), averaged across all participants × days in group", fontsize=11)
    ax.set_title(
        "Hour-of-day averaged glucose profile by study group\n"
        "(every reading binned by local hour-of-day, then averaged within each group)",
        fontsize=12.5, pad=14,
    )

    ax.legend(handles=group_lines, loc="upper left", frameon=False, fontsize=10)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    band_handles = [plt.Rectangle((0, 0), 1, 1, color=b[2], alpha=0.5) for b in bands]
    band_labels = [b[3] for b in bands]
    fig.legend(band_handles, band_labels, loc="lower center", ncol=5, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = os.path.join(HERE, "tir_tar_tbr_by_study_group.png")
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[OUT] wrote {out_path}")


if __name__ == "__main__":
    main()
