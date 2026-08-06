"""
Shared loader for insulin_dependent_k4_complications/: joins the
insulin_dependent forced-k=4 spectral clusters (cluster_matrix/
insulin_dependent/cluster_assignments_k4.csv -- see
cluster_matrix/clusters_3d_k4_insulin_dependent.py) against every
complications-adjacent variable actually available in this dataset:

  Kidney:    urine albumin, urine creatinine (-> uacr = albumin/creatinine,
             same unconverted ratio convention as clusters_9_vars/), serum
             creatinine, BUN/creatinine ratio. From final_df.csv.
  Cardiac:   NT-proBNP, Troponin T. From final_df.csv.
  Neuropathy: msslffl / mssrffl (Semmes-Weinstein monofilament foot
             sensation test, 0-10 correctly-felt sites per foot -- the
             standard diabetic peripheral neuropathy screening exam).
             NOT in final_df.csv (dataset.ipynb's cols_to_keep dropped it);
             pulled from part_meas_demo_one_hot_encoded.csv instead.

There is no retinopathy, diagnosed-nephropathy/CKD-stage, diagnosed
neuropathy, amputation, or cardiovascular-event field anywhere in this
dataset -- these six variables are the full set of complications-adjacent
measurements available, full stop.
"""
import os

import numpy as np
import pandas as pd

ROOT = "/n/groups/patel/chandrima"
CLUSTER_ASSIGNMENTS_PATH = os.path.join(ROOT, "cluster_matrix", "insulin_dependent", "cluster_assignments_k4.csv")
FINAL_DF_PATH = os.path.join(ROOT, "final_df.csv")
PART_MEAS_DEMO_PATH = os.path.join(ROOT, "part_meas_demo_one_hot_encoded.csv")

URINE_ALBUMIN_COL = "import_urine_albumin, Albumin Mass/volume in Ur"
URINE_CREATININE_COL = "import_urine_creatinine, Creatinine Mass/volume"
CREATININE_COL = "import_creatinine, Creatinine Mass/volume in Se"
BUN_CREATININE_COL = "import_buncreatinineratio, BUN/Creatinine ratio"
NT_PROBNP_COL = "import_nt_probnp, Natriuretic peptide.B prohormon"
TROPONIN_COL = "import_troponin_t, Troponin T.cardiac Mass/volum"
FOOT_LEFT_COL = "msslffl, Left Foot - Felt:"
FOOT_RIGHT_COL = "mssrffl, Right Foot - Felt:"

# Canonical marker -> display label, used consistently across plots/tables.
MARKER_LABELS = {
    "uacr": "UACR (urine albumin/creatinine ratio)",
    CREATININE_COL: "Serum creatinine",
    BUN_CREATININE_COL: "BUN/creatinine ratio",
    NT_PROBNP_COL: "NT-proBNP",
    TROPONIN_COL: "Troponin T",
    "foot_worst": "Monofilament score, worse foot (0-10)",
}
# Right-skewed markers get log1p before any parametric-adjacent summary/plot.
LOG_COLS = {"uacr", CREATININE_COL, BUN_CREATININE_COL, NT_PROBNP_COL, TROPONIN_COL}

# Which complication category each marker is a proxy for -- shown on plots
# so it's clear at a glance what's being screened for, not just the raw lab name.
MARKER_COMPLICATION = {
    "uacr": "Kidney (nephropathy)",
    CREATININE_COL: "Kidney (nephropathy)",
    BUN_CREATININE_COL: "Kidney (nephropathy)",
    NT_PROBNP_COL: "Cardiac",
    TROPONIN_COL: "Cardiac",
    "foot_worst": "Neuropathy",
}

# Same palette validated (light mode, categorical, ALL CHECKS PASS) and used
# throughout cluster_matrix/ -- kept identical here so cluster colors mean
# the same thing across both directories.
CLUSTER_PALETTE = ["#2a78d6", "#eb6834", "#2ca858", "#a259c6", "#d6b02a", "#d64550"]


def sanitize_column_name(col):
    return col.replace("[", "").replace("]", "").replace("<", "")


def load_merged():
    """Returns one row per insulin_dependent k=4 cluster participant, with
    cluster label, pc1-3, and all 6 complications markers (uacr computed)."""
    clusters = pd.read_csv(CLUSTER_ASSIGNMENTS_PATH)
    clusters["participant_id"] = clusters["participant_id"].astype(int)

    final = pd.read_csv(FINAL_DF_PATH)
    final.columns = [sanitize_column_name(c) for c in final.columns]
    final["participant_id"] = final["participant_id"].astype(int)
    lab_cols = [URINE_ALBUMIN_COL, URINE_CREATININE_COL, CREATININE_COL, BUN_CREATININE_COL, NT_PROBNP_COL, TROPONIN_COL]
    df = clusters.merge(final[["participant_id"] + lab_cols], on="participant_id", how="left")

    part_meas = pd.read_csv(PART_MEAS_DEMO_PATH)
    part_meas["participant_id"] = part_meas["participant_id"].astype(int)
    df = df.merge(part_meas[["participant_id", FOOT_LEFT_COL, FOOT_RIGHT_COL]], on="participant_id", how="left")

    df["uacr"] = df[URINE_ALBUMIN_COL] / df[URINE_CREATININE_COL].replace(0, np.nan)
    # "Worse foot" = lower (fewer correctly-felt) of the two monofilament
    # scores -- neuropathy is often asymmetric, and the worse foot is what
    # actually drives clinical risk (ADA screening uses either foot failing).
    df["foot_worst"] = df[[FOOT_LEFT_COL, FOOT_RIGHT_COL]].min(axis=1)

    return df
