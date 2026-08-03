#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cgm_ml_features.py  --  Comprehensive per-individual CGM feature extractor
===========================================================================

Purpose
-------
Turn each participant's Dexcom G6 continuous-glucose-monitoring (CGM) time
series into ONE ROW of numeric features suitable as input to a machine
-learning pipeline. This goes well *beyond* time-in-range / time-above /
time-below and covers the validated glycemic-variability, risk, dynamics,
signal-complexity, spectral, and circadian metric families from the CGM
literature (see the companion table CGM_FEATURES.md for a metric-by-metric
citation list).

Source data
-----------
Reads from ai_readi/preprocessed/cgm.parquet (participant_id, glucose,
start_time_local, ...), which already reconstructs each reading's local wall
-clock time per participant's own recorded timezone (America/Los_Angeles or
America/Chicago, DST-aware) -- see additional_extracted_CGM.ipynb for the
original parsing of this parquet from the raw Dexcom JSON export. Metrics are
computed from those local timestamps directly.

What it produces
----------------
  cgm_ml_features.csv             one row per participant_id, ~70 numeric cols
  cgm_ml_features_dictionary.csv  column -> description, units, citation-key

Usage
-----
  # 1. standard run against ai_readi/preprocessed/cgm.parquet:
  python3 cgm_ml_features.py

  # 2. point at a specific parquet location:
  python3 cgm_ml_features.py --cgm-parquet /path/to/cgm.parquet --out cgm_ml_features.csv

  # 3. also merge the features into final_df.csv (drops the legacy
  #    tir/tar/tbr/cv_glucose/mean_of_nightly_means/mage/modd columns first):
  python3 cgm_ml_features.py --merge final_df.csv

  # 4. verify the code works WITHOUT the real data (synthetic self-test):
  python3 cgm_ml_features.py --selftest

Dependencies: numpy, pandas, pyarrow (scipy is used if present but NOT required).

Notes / caveats
---------------
* Features are computed on a regular 5-min grid reconstructed per participant;
  gaps are left as NaN and every metric is gap-aware.
* Metrics that need >= a few days (MODD, between-day SD, cosinor, spectral)
  return NaN when wear is too short -- always QC on `qc_pct_active` /
  `qc_n_days` before modeling. International consensus = >=14 days & >=70%
  active (Battelino 2019).
* Many variability indices are strongly collinear (SD, CV, J-index, GRADE,
  M-value, HBGI all track mean/hyperglycemia). See CGM_FEATURES.md
  "redundancy" notes -- this pipeline ships the full family list and leaves
  selection to ElasticNetCV / XGBoost importance downstream.
"""
from __future__ import annotations
import os, sys, argparse
import datetime as dt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration (defaults; override on the CLI)
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CGM_PARQUET = os.path.join(HERE, "ai_readi", "preprocessed", "cgm.parquet")

INTERVAL_MIN = 5                         # Dexcom G6 cadence
PER_DAY = 24 * 60 // INTERVAL_MIN        # 288 readings/day at full wear
MIN_READINGS = 48                        # need ~4h of data to bother

# glucose thresholds (mg/dL) -- Battelino 2019 international CGM consensus
V_LOW, LOW, HIGH, V_HIGH = 54, 70, 180, 250
TIGHT_LO, TIGHT_HI = 70, 140

# Legacy CGM columns produced by the earlier, narrower feature scripts
# (additional_extracted_CGM.ipynb); dropped from final_df.csv on --merge in
# favor of the superset this script emits (mage/modd are recomputed here
# under the same names, so they're replaced rather than duplicated).
LEGACY_CGM_COLS = ["tir", "tar", "tbr", "cv_glucose", "mean_of_nightly_means"]


# ===========================================================================
# Parsing
# ===========================================================================
def load_all(cgm_parquet: str) -> pd.DataFrame:
    """Read the preprocessed CGM parquet -> DataFrame with participant_id,
    glucose (mg/dL), start_time_local (tz-naive local wall clock)."""
    df = pd.read_parquet(cgm_parquet, columns=[
        "participant_id", "glucose", "start_time_local", "event_type",
    ])
    df = df[df["event_type"] == "EGV"]
    df = df.dropna(subset=["glucose", "start_time_local"])
    df["start_time_local"] = pd.to_datetime(df["start_time_local"]).dt.tz_localize(None)
    return df[["participant_id", "glucose", "start_time_local"]]


def load_series(sub: pd.DataFrame):
    """One participant's rows -> (values[mg/dL], local_datetimes) sorted by time."""
    sub = sub.sort_values("start_time_local")
    vals = sub["glucose"].to_numpy(dtype=float)
    times = sub["start_time_local"].dt.to_pydatetime()
    return vals, np.asarray(times)


def regular_grid(vals, times):
    """Snap readings onto a regular 5-min grid over the wear span.
    Returns (grid_glucose_with_nan, grid_datetimes). Enables MODD/CONGA/FFT/
    entropy which assume even sampling."""
    if len(vals) < 2:
        return vals, times
    t0, t1 = times.min(), times.max()
    n = int((t1 - t0).total_seconds() // (INTERVAL_MIN * 60)) + 1
    grid_t = np.array([t0 + dt.timedelta(minutes=INTERVAL_MIN * k) for k in range(n)])
    g = np.full(n, np.nan)
    idx = np.round((np.array([(t - t0).total_seconds() for t in times])
                    / (INTERVAL_MIN * 60))).astype(int)
    ok = (idx >= 0) & (idx < n)
    g[idx[ok]] = vals[ok]                            # last write wins on collision
    return g, grid_t


# ===========================================================================
# Metric helpers
# ===========================================================================
def _finite(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def bg_risk(bg):
    """Kovatchev symmetric BG-risk transform -> (low, high) risk arrays.
    Basis of LBGI / HBGI / ADRR (Kovatchev 1997, 1998, 2006)."""
    bg = np.clip(np.asarray(bg, float), 20, 600)
    f = 1.509 * (np.log(bg) ** 1.084 - 5.381)
    rl = np.where(f < 0, 10 * f ** 2, 0.0)
    rh = np.where(f > 0, 10 * f ** 2, 0.0)
    return rl, rh


def mage(g, sd):
    """Mean Amplitude of Glycemic Excursions (Service 1970): mean turning-point
    amplitude exceeding 1 SD of the series (simplified Service algorithm)."""
    g = _finite(g)
    if len(g) < 3 or not np.isfinite(sd) or sd <= 0:
        return np.nan
    dd = np.diff(g)
    tp = [0]
    for i in range(1, len(dd)):
        if dd[i - 1] != 0 and dd[i] != 0 and np.sign(dd[i]) != np.sign(dd[i - 1]):
            tp.append(i)
    tp.append(len(g) - 1)
    amps = np.abs(np.diff(g[tp]))
    valid = amps[amps > sd]
    return float(np.mean(valid)) if len(valid) else 0.0


def conga(g_grid, hours):
    """CONGA-n (McDonnell 2005): SD of differences between readings n hours
    apart on the regular grid."""
    lag = int(hours * 60 // INTERVAL_MIN)
    if lag <= 0 or lag >= len(g_grid):
        return np.nan
    d = g_grid[lag:] - g_grid[:-lag]
    d = d[np.isfinite(d)]
    return float(np.std(d, ddof=1)) if len(d) > 1 else np.nan


def modd(g_grid, grid_t):
    """MODD (Molnar 1972): mean of |glucose(t) - glucose(t - 24h)| over the grid."""
    lag = PER_DAY
    if lag >= len(g_grid):
        return np.nan
    d = np.abs(g_grid[lag:] - g_grid[:-lag])
    d = d[np.isfinite(d)]
    return float(np.mean(d)) if len(d) else np.nan


def j_index(mean, sd):
    """J-index (Wojcicki 1995): 0.001*(mean+SD)^2, mg/dL units."""
    if not (np.isfinite(mean) and np.isfinite(sd)):
        return np.nan
    return 0.001 * (mean + sd) ** 2


def m_value(g, target=120.0):
    """M-value (Schlichtkrull 1965) around a target (default 120 mg/dL)."""
    g = _finite(g)
    if len(g) == 0:
        return np.nan
    return float(np.mean(np.abs(10 * np.log10(g / target)) ** 3))


def grade(g):
    """GRADE (Hill 2007): mean glycemic-risk score; mg/dL converted to mmol/L."""
    g = _finite(g) / 18.0                            # mg/dL -> mmol/L
    if len(g) == 0:
        return np.nan
    h = 425 * (np.log10(np.log10(np.clip(g, 1e-3, None))) + 0.16) ** 2
    return float(np.mean(np.clip(h, 0, 50)))


def mag(g_grid, grid_t):
    """MAG (Hermanides 2010): total absolute glucose change per hour, over
    consecutive valid readings."""
    ok = np.isfinite(g_grid)
    gv = g_grid[ok]
    if len(gv) < 2:
        return np.nan
    tt = grid_t[ok]
    hours = (tt[-1] - tt[0]).total_seconds() / 3600.0
    if hours <= 0:
        return np.nan
    return float(np.sum(np.abs(np.diff(gv))) / hours)


def sample_entropy(x, m=2, r=0.2):
    """Sample entropy (Richman & Moorman 2000); r as fraction of SD.
    Vectorized Chebyshev-distance neighbor count. NaNs dropped (breaks
    templates across gaps, acceptable for a global complexity summary)."""
    x = _finite(x)
    N = len(x)
    if N < m + 2:
        return np.nan
    sd = np.std(x)
    if sd == 0:
        return np.nan
    tol = r * sd

    def _phi(mm):
        tmpl = np.array([x[i:i + mm] for i in range(N - mm + 1)])
        cnt = 0
        for i in range(len(tmpl)):
            d = np.max(np.abs(tmpl - tmpl[i]), axis=1)
            cnt += np.count_nonzero(d <= tol) - 1     # exclude self-match
        return cnt

    B = _phi(m)
    A = _phi(m + 1)
    if B == 0 or A == 0:
        return np.nan
    return float(-np.log(A / B))


def dfa_alpha(x, scales=(4, 8, 16, 32, 64)):
    """Detrended Fluctuation Analysis exponent (Peng 1994). alpha ~0.5 random,
    ~1 long-range correlated, >1 nonstationary."""
    x = _finite(x)
    if len(x) < max(scales) * 2:
        return np.nan
    y = np.cumsum(x - x.mean())
    F, S = [], []
    for s in scales:
        nseg = len(y) // s
        if nseg < 1:
            continue
        rms = []
        t = np.arange(s)
        for v in range(nseg):
            seg = y[v * s:(v + 1) * s]
            fit = np.polyval(np.polyfit(t, seg, 1), t)
            rms.append(np.sqrt(np.mean((seg - fit) ** 2)))
        f = np.mean(rms)
        if f > 0:
            F.append(np.log(f)); S.append(np.log(s))
    if len(S) < 2:
        return np.nan
    return float(np.polyfit(S, F, 1)[0])


def poincare(g_grid):
    """Poincare plot descriptors on lag-1 (SD1 short-term, SD2 long-term)."""
    g = g_grid
    ok = np.isfinite(g[:-1]) & np.isfinite(g[1:])
    x1, x2 = g[:-1][ok], g[1:][ok]
    if len(x1) < 3:
        return np.nan, np.nan, np.nan
    diff = (x2 - x1) / np.sqrt(2)
    summ = (x2 + x1) / np.sqrt(2)
    sd1 = float(np.std(diff, ddof=1))
    sd2 = float(np.std(summ, ddof=1))
    ratio = sd1 / sd2 if sd2 else np.nan
    return sd1, sd2, ratio


def autocorr1(g_grid):
    """Lag-1 autocorrelation on the regular grid (persistence of glucose)."""
    ok = np.isfinite(g_grid[:-1]) & np.isfinite(g_grid[1:])
    a, b = g_grid[:-1][ok], g_grid[1:][ok]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def spectral_features(g_grid):
    """FFT on a mean-imputed grid -> (circadian-band power fraction, dominant
    period in hours, spectral entropy). Captures periodicity beyond time-domain
    variability."""
    g = g_grid.copy()
    ok = np.isfinite(g)
    if ok.sum() < PER_DAY * 2:                       # need >= ~2 days
        return np.nan, np.nan, np.nan
    g[~ok] = np.nanmean(g)                           # mean-impute gaps
    g = g - g.mean()
    n = len(g)
    freqs = np.fft.rfftfreq(n, d=INTERVAL_MIN / 60.0)   # cycles per hour
    power = np.abs(np.fft.rfft(g)) ** 2
    power[0] = 0.0
    total = power.sum()
    if total <= 0:
        return np.nan, np.nan, np.nan
    # circadian band ~ 20-28 h period
    band = (freqs >= 1 / 28.0) & (freqs <= 1 / 20.0)
    circ_frac = float(power[band].sum() / total)
    dom_period = float(1.0 / freqs[1:][np.argmax(power[1:])]) if n > 2 else np.nan
    p = power[1:] / power[1:].sum()
    p = p[p > 0]
    spec_ent = float(-np.sum(p * np.log(p)) / np.log(len(p))) if len(p) > 1 else np.nan
    return circ_frac, dom_period, spec_ent


def cosinor(vals, times):
    """Single-component 24h cosinor (Cornelissen 2014): least-squares fit of
    MESOR + amplitude*cos(2*pi*t/24 - acrophase). t = hours-of-day."""
    v = _finite(vals)
    if len(v) < PER_DAY:
        return np.nan, np.nan, np.nan
    hod = np.array([t.hour + t.minute / 60.0 for t in times])[np.isfinite(vals)]
    w = 2 * np.pi / 24.0
    X = np.column_stack([np.ones_like(hod), np.cos(w * hod), np.sin(w * hod)])
    try:
        beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    except Exception:
        return np.nan, np.nan, np.nan
    mesor, bcos, bsin = beta
    amp = float(np.hypot(bcos, bsin))
    # v = mesor + bcos*cos(w*hod) + bsin*sin(w*hod) = mesor + amp*cos(w*(hod - acro))
    # requires bcos = amp*cos(w*acro), bsin = amp*sin(w*acro) -> w*acro = atan2(bsin, bcos).
    # (Previously used atan2(-bsin, bcos), which returns the 24h-complement of the
    # true peak hour, not the peak hour itself -- confirmed with a synthetic signal
    # constructed to peak at hour 8, which this line used to report as acro=16.)
    acro = float((np.arctan2(bsin, bcos) % (2 * np.pi)) * 24 / (2 * np.pi))  # peak hour
    return float(mesor), amp, acro


def episodes(g, thresh, below=True, min_len=3):
    """Count/duration of hypo (below) or hyper (above) episodes:
    >= min_len consecutive readings (>=15 min) past threshold."""
    g = _finite(g)
    flag = (g < thresh) if below else (g > thresh)
    n_ev, run, durations = 0, 0, []
    for b in flag:
        if b:
            run += 1
        else:
            if run >= min_len:
                n_ev += 1; durations.append(run * INTERVAL_MIN)
            run = 0
    if run >= min_len:
        n_ev += 1; durations.append(run * INTERVAL_MIN)
    mean_dur = float(np.mean(durations)) if durations else 0.0
    return n_ev, mean_dur


# ===========================================================================
# Per-participant feature vector
# ===========================================================================
def extract(vals, times):
    """All features for one participant. Returns {} if too little data."""
    if len(vals) < MIN_READINGS:
        return {}
    g_grid, grid_t = regular_grid(vals, times)
    a = _finite(vals)
    n = len(a)
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if n > 1 else np.nan
    cv = 100 * sd / mean if mean else np.nan

    hod = np.array([t.hour + t.minute / 60.0 for t in times])
    day = np.array([t.date() for t in times])
    overnight = a[(hod >= 0) & (hod < 6)]
    daytime = a[hod >= 6]
    dawn = a[(hod >= 6) & (hod < 9)]

    # ---- per-day aggregates (within/between-day variability) ----------------
    dmeans, dranges, dhyperauc, dhypoauc = [], [], [], []
    for u in np.unique(day):
        gd = a[day == u]
        if len(gd) >= 12:                            # >= ~1h that day
            dmeans.append(gd.mean())
            dranges.append(gd.max() - gd.min())
            dhyperauc.append(np.sum(np.clip(gd - HIGH, 0, None)) * INTERVAL_MIN / 60.0)
            dhypoauc.append(np.sum(np.clip(LOW - gd, 0, None)) * INTERVAL_MIN / 60.0)

    # ---- risk transforms ----------------------------------------------------
    rl, rh = bg_risk(a)
    lr_hr = []
    for u in np.unique(day):
        idx = day == u
        if idx.sum() >= 12:
            lr_hr.append(rl[idx].max() + rh[idx].max())

    # ---- rate of change (per grid step, mg/dL/min) --------------------------
    dg = np.diff(g_grid)
    dt_min = INTERVAL_MIN
    roc = dg[np.isfinite(dg)] / dt_min
    sd1, sd2, sd_ratio = poincare(g_grid)
    circ_frac, dom_period, spec_ent = spectral_features(g_grid)
    mesor, cos_amp, acrophase = cosinor(vals, times)

    # ---- GRI (Klonoff 2022) -------------------------------------------------
    p_vlow = 100 * np.mean(a < V_LOW)
    p_low = 100 * np.mean((a >= V_LOW) & (a < LOW))
    p_vhigh = 100 * np.mean(a > V_HIGH)
    p_high = 100 * np.mean((a > HIGH) & (a <= V_HIGH))
    hypo_c = p_vlow + 0.8 * p_low
    hyper_c = p_vhigh + 0.5 * p_high

    n_hypo, dur_hypo = episodes(a, LOW, below=True)
    n_hyper, dur_hyper = episodes(a, HIGH, below=False)

    span_days = (times.max() - times.min()).total_seconds() / 86400.0
    f = {
        # -- QC / coverage --
        "qc_n_egv": n,
        "qc_wear_span_days": round(span_days, 2),
        "qc_n_days": int(len(np.unique(day))),
        "qc_pct_active": round(100 * n / (span_days * PER_DAY), 1) if span_days > 0 else np.nan,

        # -- central tendency & distribution shape --
        "mean_glucose": round(mean, 1),
        "sd_glucose": round(sd, 1),
        "cv_pct": round(cv, 1),
        "iqr_glucose": round(float(np.percentile(a, 75) - np.percentile(a, 25)), 1),
        "mad_glucose": round(float(np.median(np.abs(a - np.median(a)))), 1),
        "min_glucose": float(a.min()),
        "max_glucose": float(a.max()),
        "range_glucose": float(a.max() - a.min()),
        "p10_glucose": round(float(np.percentile(a, 10)), 1),
        "p25_glucose": round(float(np.percentile(a, 25)), 1),
        "median_glucose": round(float(np.median(a)), 1),
        "p75_glucose": round(float(np.percentile(a, 75)), 1),
        "p90_glucose": round(float(np.percentile(a, 90)), 1),
        "skewness": round(_skew(a), 3),
        "kurtosis": round(_kurt(a), 3),

        # -- time-in-range family (reference; the point is to go beyond these) --
        "tir_70_180_pct": round(100 * np.mean((a >= LOW) & (a <= HIGH)), 1),
        "tbr_lt70_pct": round(100 * np.mean(a < LOW), 1),
        "tbr_lt54_pct": round(100 * np.mean(a < V_LOW), 1),
        "tar_gt180_pct": round(100 * np.mean(a > HIGH), 1),
        "tar_gt250_pct": round(100 * np.mean(a > V_HIGH), 1),
        "tight_70_140_pct": round(100 * np.mean((a >= TIGHT_LO) & (a <= TIGHT_HI)), 1),

        # -- glycemic-variability indices --
        "gmi_pct": round(3.31 + 0.02392 * mean, 2),
        "mage": round(mage(a, sd), 1),
        "modd": round(modd(g_grid, grid_t), 1),
        "conga1": round(conga(g_grid, 1), 1),
        "conga2": round(conga(g_grid, 2), 1),
        "conga4": round(conga(g_grid, 4), 1),
        "conga6": round(conga(g_grid, 6), 1),
        "j_index": round(j_index(mean, sd), 1),
        "m_value": round(m_value(a), 1),
        "grade": round(grade(a), 1),
        "mag_per_hour": round(mag(g_grid, grid_t), 1),
        "mean_daily_range": round(float(np.mean(dranges)), 1) if dranges else np.nan,
        "within_day_sd": round(float(np.mean([np.std(a[day == u], ddof=1)
                              for u in np.unique(day) if (day == u).sum() > 1])), 1)
                              if len(dmeans) else np.nan,
        "between_day_sd": round(float(np.std(dmeans, ddof=1)), 1) if len(dmeans) > 1 else np.nan,

        # -- risk indices --
        "lbgi": round(float(np.mean(rl)), 2),
        "hbgi": round(float(np.mean(rh)), 2),
        "adrr": round(float(np.mean(lr_hr)), 1) if lr_hr else np.nan,
        "gri": round(float(min(3.0 * hypo_c + 1.6 * hyper_c, 100.0)), 1),
        "gri_hypo_comp": round(float(hypo_c), 1),
        "gri_hyper_comp": round(float(hyper_c), 1),

        # -- dynamics / rate of change --
        "roc_mean_abs": round(float(np.mean(np.abs(roc))), 3) if len(roc) else np.nan,
        "roc_sd": round(float(np.std(roc, ddof=1)), 3) if len(roc) > 1 else np.nan,
        "pct_rapid_rise": round(100 * np.mean(roc > 2), 1) if len(roc) else np.nan,
        "pct_rapid_fall": round(100 * np.mean(roc < -2), 1) if len(roc) else np.nan,

        # -- excursion / episode burden --
        "n_hypo_events": n_hypo,
        "mean_hypo_dur_min": round(dur_hypo, 1),
        "n_hyper_events": n_hyper,
        "mean_hyper_dur_min": round(dur_hyper, 1),
        "hyper_auc_per_day": round(float(np.mean(dhyperauc)), 1) if dhyperauc else np.nan,
        "hypo_auc_per_day": round(float(np.mean(dhypoauc)), 2) if dhypoauc else np.nan,

        # -- signal complexity / nonlinear dynamics --
        "sample_entropy": round(sample_entropy(g_grid), 3),
        "dfa_alpha": round(dfa_alpha(g_grid), 3),
        "poincare_sd1": round(sd1, 1) if np.isfinite(sd1) else np.nan,
        "poincare_sd2": round(sd2, 1) if np.isfinite(sd2) else np.nan,
        "poincare_ratio": round(sd_ratio, 3) if np.isfinite(sd_ratio) else np.nan,
        "autocorr_lag1": round(autocorr1(g_grid), 3),

        # -- frequency domain --
        "spectral_circadian_frac": round(circ_frac, 3) if np.isfinite(circ_frac) else np.nan,
        "spectral_dominant_period_h": round(dom_period, 1) if np.isfinite(dom_period) else np.nan,
        "spectral_entropy": round(spec_ent, 3) if np.isfinite(spec_ent) else np.nan,

        # -- circadian / temporal pattern --
        "cosinor_mesor": round(mesor, 1) if np.isfinite(mesor) else np.nan,
        "cosinor_amplitude": round(cos_amp, 1) if np.isfinite(cos_amp) else np.nan,
        "cosinor_acrophase_h": round(acrophase, 1) if np.isfinite(acrophase) else np.nan,
        "overnight_mean": round(float(overnight.mean()), 1) if len(overnight) else np.nan,
        "daytime_mean": round(float(daytime.mean()), 1) if len(daytime) else np.nan,
        "dawn_rise": (round(float(dawn.mean() - overnight.mean()), 1)
                      if len(dawn) and len(overnight) else np.nan),
        "nocturnal_hypo_pct": round(100 * np.mean(overnight < LOW), 2) if len(overnight) else np.nan,
    }
    return f


def _skew(a):
    a = np.asarray(a, float); m = a.mean(); s = a.std()
    return float(np.mean(((a - m) / s) ** 3)) if s > 0 else np.nan


def _kurt(a):
    a = np.asarray(a, float); m = a.mean(); s = a.std()
    return float(np.mean(((a - m) / s) ** 4) - 3) if s > 0 else np.nan


# ===========================================================================
# Data dictionary (column -> description, units, citation key -> CGM_FEATURES.md)
# ===========================================================================
DICTIONARY = [
    ("qc_n_egv", "Number of valid EGV readings.", "count", "Battelino2019"),
    ("qc_wear_span_days", "Days between first and last reading.", "days", "Battelino2019"),
    ("qc_n_days", "Distinct calendar days with data.", "days", "Battelino2019"),
    ("qc_pct_active", "% of expected 5-min readings present.", "%", "Battelino2019"),
    ("mean_glucose", "Mean sensor glucose.", "mg/dL", "Rodbard2009"),
    ("sd_glucose", "SD of glucose.", "mg/dL", "Rodbard2009"),
    ("cv_pct", "Coefficient of variation (100*SD/mean); >=36% unstable.", "%", "Rodbard2009"),
    ("iqr_glucose", "Interquartile range of glucose.", "mg/dL", "Rodbard2009"),
    ("mad_glucose", "Median absolute deviation.", "mg/dL", "Rodbard2009"),
    ("min_glucose", "Minimum glucose.", "mg/dL", "Battelino2019"),
    ("max_glucose", "Maximum glucose.", "mg/dL", "Battelino2019"),
    ("range_glucose", "Max - min glucose.", "mg/dL", "Rodbard2009"),
    ("p10_glucose", "10th percentile glucose.", "mg/dL", "Rodbard2009"),
    ("p25_glucose", "25th percentile glucose.", "mg/dL", "Rodbard2009"),
    ("median_glucose", "Median glucose.", "mg/dL", "Rodbard2009"),
    ("p75_glucose", "75th percentile glucose.", "mg/dL", "Rodbard2009"),
    ("p90_glucose", "90th percentile glucose.", "mg/dL", "Rodbard2009"),
    ("skewness", "Skewness of glucose distribution.", "unitless", "Rodbard2009"),
    ("kurtosis", "Excess kurtosis of glucose distribution.", "unitless", "Rodbard2009"),
    ("tir_70_180_pct", "Time in range 70-180.", "%", "Battelino2019"),
    ("tbr_lt70_pct", "Time below 70.", "%", "Battelino2019"),
    ("tbr_lt54_pct", "Time below 54.", "%", "Battelino2019"),
    ("tar_gt180_pct", "Time above 180.", "%", "Battelino2019"),
    ("tar_gt250_pct", "Time above 250.", "%", "Battelino2019"),
    ("tight_70_140_pct", "Time in tight range 70-140.", "%", "Battelino2019"),
    ("gmi_pct", "Glucose Management Indicator.", "%", "Bergenstal2018"),
    ("mage", "Mean Amplitude of Glycemic Excursions (>1 SD).", "mg/dL", "Service1970"),
    ("modd", "Mean Of Daily Differences (24h).", "mg/dL", "Molnar1972"),
    ("conga1", "CONGA over 1 h.", "mg/dL", "McDonnell2005"),
    ("conga2", "CONGA over 2 h.", "mg/dL", "McDonnell2005"),
    ("conga4", "CONGA over 4 h.", "mg/dL", "McDonnell2005"),
    ("conga6", "CONGA over 6 h.", "mg/dL", "McDonnell2005"),
    ("j_index", "J-index = 0.001*(mean+SD)^2.", "unitless", "Wojcicki1995"),
    ("m_value", "M-value around 120 mg/dL target.", "unitless", "Schlichtkrull1965"),
    ("grade", "Glycemic Risk Assessment Diabetes Equation, mean score.", "unitless", "Hill2007"),
    ("mag_per_hour", "Mean Absolute Glucose change per hour.", "mg/dL/h", "Hermanides2010"),
    ("mean_daily_range", "Mean of (daily max - daily min).", "mg/dL", "Rodbard2009"),
    ("within_day_sd", "Mean within-day SD (SDw).", "mg/dL", "Rodbard2009"),
    ("between_day_sd", "SD of per-day means (SDb).", "mg/dL", "Rodbard2009"),
    ("lbgi", "Low Blood Glucose Index.", "index", "Kovatchev1997"),
    ("hbgi", "High Blood Glucose Index.", "index", "Kovatchev1998"),
    ("adrr", "Average Daily Risk Range.", "index", "Kovatchev2006"),
    ("gri", "Glycemia Risk Index (0-100).", "0-100", "Klonoff2022"),
    ("gri_hypo_comp", "GRI hypoglycemia component.", "index", "Klonoff2022"),
    ("gri_hyper_comp", "GRI hyperglycemia component.", "index", "Klonoff2022"),
    ("roc_mean_abs", "Mean absolute rate of change.", "mg/dL/min", "Rodbard2009"),
    ("roc_sd", "SD of rate of change.", "mg/dL/min", "Rodbard2009"),
    ("pct_rapid_rise", "% steps rising >2 mg/dL/min.", "%", "Rodbard2009"),
    ("pct_rapid_fall", "% steps falling >2 mg/dL/min.", "%", "Rodbard2009"),
    ("n_hypo_events", "# hypo episodes (>=15 min <70).", "count", "Battelino2019"),
    ("mean_hypo_dur_min", "Mean hypo episode duration.", "min", "Battelino2019"),
    ("n_hyper_events", "# hyper episodes (>=15 min >180).", "count", "Battelino2019"),
    ("mean_hyper_dur_min", "Mean hyper episode duration.", "min", "Battelino2019"),
    ("hyper_auc_per_day", "Mean daily AUC above 180.", "mg/dL*h/day", "Rodbard2009"),
    ("hypo_auc_per_day", "Mean daily AUC below 70.", "mg/dL*h/day", "Rodbard2009"),
    ("sample_entropy", "Sample entropy (m=2, r=0.2SD); signal regularity.", "nats", "Richman2000/Crenier2016"),
    ("dfa_alpha", "Detrended fluctuation exponent.", "unitless", "Peng1995/Signal2013"),
    ("poincare_sd1", "Poincare short-term variability.", "mg/dL", "Crenier2014"),
    ("poincare_sd2", "Poincare long-term variability.", "mg/dL", "Crenier2014"),
    ("poincare_ratio", "SD1/SD2.", "unitless", "Crenier2014"),
    ("autocorr_lag1", "Lag-1 autocorrelation (5 min).", "unitless", "Rodbard2009"),
    ("spectral_circadian_frac", "Power fraction in 20-28h band.", "fraction", "Rodbard2009"),
    ("spectral_dominant_period_h", "Dominant FFT period.", "h", "Rodbard2009"),
    ("spectral_entropy", "Normalized spectral entropy.", "unitless", "Rodbard2009"),
    ("cosinor_mesor", "Cosinor rhythm-adjusted mean.", "mg/dL", "Cornelissen2014"),
    ("cosinor_amplitude", "Cosinor 24h amplitude.", "mg/dL", "Cornelissen2014"),
    ("cosinor_acrophase_h", "Cosinor peak hour (0-24).", "h", "Cornelissen2014"),
    ("overnight_mean", "Mean glucose 00-06h.", "mg/dL", "Rodbard2009"),
    ("daytime_mean", "Mean glucose 06-24h.", "mg/dL", "Rodbard2009"),
    ("dawn_rise", "Dawn effect: mean(06-09h) - mean(00-06h).", "mg/dL", "Rodbard2009"),
    ("nocturnal_hypo_pct", "% overnight readings <70.", "%", "Battelino2019"),
]


# ===========================================================================
# Driver
# ===========================================================================
def run(cgm_parquet, out_csv, dict_csv, merge_csv=None):
    print(f"[CGM-ML] loading {cgm_parquet}")
    cgm = load_all(cgm_parquet)
    pids = sorted(cgm["participant_id"].unique())
    print(f"[CGM-ML] {len(pids)} participants in the CGM parquet")
    rows = []
    for i, (pid, sub) in enumerate(cgm.groupby("participant_id"), 1):
        try:
            vals, times = load_series(sub)
            feat = extract(vals, times)
        except Exception as e:
            print(f"   [warn] {pid}: {e}")
            continue
        if feat:
            feat["person_id"] = int(pid) if str(pid).isdigit() else pid
            rows.append(feat)
        if i % 250 == 0:
            print(f"   ... {i}/{len(pids)}  ({len(rows)} with features)")

    if not rows:
        print("[CGM-ML] no features extracted -- check --cgm-parquet.")
        return
    feat = pd.DataFrame(rows)
    cols = ["person_id"] + [c for c in feat.columns if c != "person_id"]
    feat = feat[cols]
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    feat.to_csv(out_csv, index=False)
    print(f"[OUT] wrote {out_csv}  ({feat.shape[0]} participants x {feat.shape[1]-1} features)")

    dd = pd.DataFrame(DICTIONARY, columns=["column", "description", "units", "citation_key"])
    dd.to_csv(dict_csv, index=False)
    print(f"[OUT] wrote {dict_csv}  ({len(dd)} features documented)")

    if merge_csv:
        base = pd.read_csv(merge_csv)
        base = base.rename(columns={"person_id": "participant_id"}) if "person_id" in base.columns else base
        feat_merge = feat.rename(columns={"person_id": "participant_id"})
        newcols = [c for c in feat_merge.columns if c != "participant_id"]
        drop_cols = set(newcols) | (set(LEGACY_CGM_COLS) & set(base.columns))
        base = base.drop(columns=[c for c in drop_cols if c in base.columns], errors="ignore")
        base = base.merge(feat_merge, on="participant_id", how="left")
        base.to_csv(merge_csv, index=False)
        print(f"[OUT] merged {len(newcols)} features into {merge_csv} -> {base.shape[1]} cols "
              f"(dropped legacy cols: {sorted(set(LEGACY_CGM_COLS) & drop_cols)})")


# ---------------------------------------------------------------------------
# Synthetic self-test (verify the code runs without the real dataset)
# ---------------------------------------------------------------------------
def selftest():
    print("[SELFTEST] generating 10 days of synthetic 5-min CGM ...")
    rng = np.random.default_rng(0)
    n = PER_DAY * 10
    t0 = dt.datetime(2024, 1, 1)
    times = np.array([t0 + dt.timedelta(minutes=5 * k) for k in range(n)])
    hod = np.array([t.hour + t.minute / 60 for t in times])
    base = 120 + 30 * np.sin(2 * np.pi * (hod - 6) / 24)          # circadian
    meals = 40 * np.exp(-((hod - 8) ** 2) / 2) + 45 * np.exp(-((hod - 13) ** 2) / 2) \
            + 50 * np.exp(-((hod - 19) ** 2) / 1.5)               # meal spikes
    vals = base + meals + rng.normal(0, 8, n)
    vals = np.clip(vals, 40, 400)
    mask = rng.random(n) > 0.05                                   # 5% dropout
    feat = extract(vals[mask], times[mask])
    print(f"[SELFTEST] extracted {len(feat)} features:")
    for k in sorted(feat):
        print(f"   {k:32s} = {feat[k]}")
    assert feat and np.isfinite(feat["sample_entropy"]) and np.isfinite(feat["mage"]), \
        "self-test failed"
    print("[SELFTEST] OK -- all families computed.")


def main():
    ap = argparse.ArgumentParser(description="Extract per-individual CGM ML features.")
    ap.add_argument("--cgm-parquet", default=DEFAULT_CGM_PARQUET, help="Preprocessed CGM parquet.")
    ap.add_argument("--out", default=os.path.join(HERE, "cgm_ml_features.csv"))
    ap.add_argument("--dict", default=os.path.join(HERE, "cgm_ml_features_dictionary.csv"))
    ap.add_argument("--merge", default=None, help="Optional analysis CSV to merge features into.")
    ap.add_argument("--selftest", action="store_true", help="Run on synthetic data and exit.")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if not os.path.isfile(args.cgm_parquet):
        sys.exit(f"[ERROR] --cgm-parquet not found: {args.cgm_parquet}\n"
                 f"        Run 'python3 {sys.argv[0]} --selftest' to test the code locally.")
    run(args.cgm_parquet, args.out, args.dict, args.merge)


if __name__ == "__main__":
    main()
