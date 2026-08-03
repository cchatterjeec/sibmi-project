"""
Shared constants/helpers for the study-group-stratified rerun of clusters_9_vars/.

Same 9-variable diabetes-relevant panel (no HbA1c/glucose), same preprocessing
(log1p on the 6 skewed labs, StandardScaler), same k-selection-by-silhouette
approach -- just refit separately within each study_group stratum instead of
pooling the cohort. Rationale: the pooled cluster's kidney-complication group
was already enriched 42.6% insulin_dependent, i.e. partly reconstructing
diabetes status. Stratifying asks the sharper question: does a comparable
kidney/lipid/inflammation sub-phenotype -- and CGM's ability to predict it --
still show up *within* a single clinical stratum, where it can't just be
rediscovering who has diabetes.

GROUPS maps a short slug (used for the output subdirectory and CLI --group
argument) to the actual study_group string in final_df.csv.
"""
import json
import os
import re

import numpy as np
import pandas as pd

ROOT = "/n/groups/patel/chandrima"
DATA_PATH = os.path.join(ROOT, "final_df.csv")
BY_GROUP_ROOT = os.path.join(ROOT, "clusters_9_vars_by_group")
CGM_QC_MIN_PCT_ACTIVE = 70.0
K_RANGE = range(2, 9)
VIEW_ELEV, VIEW_AZIM = 20, 45
CLUSTER_PALETTE = ["#2a78d6", "#eb6834", "#2ca858", "#a259c6", "#d6b02a", "#d64550"]

GROUPS = {
    "healthy": "healthy",
    "pre_diabetes": "pre_diabetes_lifestyle_controlled",
    "oral_medication": "oral_medication_and_or_non_insulin_injectable_medication_controlled",
    "insulin_dependent": "insulin_dependent",
}


def sanitize_column_name(col):
    return re.sub(r"[\[\]<]", "", col)


CPEPTIDE_COL = sanitize_column_name("import_c_peptide, C peptide [Mass/volume] in Seru")
INSULIN_COL = sanitize_column_name("import_insulin, Insulin [Units/volume] in Serum o")
BMI_COL = "bmi_vsorres, BMI"
CREATININE_COL = sanitize_column_name("import_creatinine, Creatinine [Mass/volume] in Se")
URINE_ALBUMIN_COL = sanitize_column_name("import_urine_albumin, Albumin [Mass/volume] in Ur")
URINE_CREATININE_COL = sanitize_column_name("import_urine_creatinine, Creatinine [Mass/volume]")
TRIGLYCERIDES_COL = sanitize_column_name("import_triglycerides, Triglyceride [Mass/volume] ")
HDL_COL = sanitize_column_name("import_hdl_cholesterol, Cholesterol in HDL [Mass/")
CRP_COL = sanitize_column_name("import_crp_hs, C reactive protein [Mass/volume] i")

VAR_LABELS = {
    "age": "Age", BMI_COL: "BMI", INSULIN_COL: "Insulin", CPEPTIDE_COL: "C-peptide",
    CREATININE_COL: "Creatinine", "uacr": "UACR", TRIGLYCERIDES_COL: "Triglycerides",
    HDL_COL: "HDL", CRP_COL: "CRP",
}
FEATURE_COLS = ["age", BMI_COL, INSULIN_COL, CPEPTIDE_COL, CREATININE_COL, "uacr", TRIGLYCERIDES_COL, HDL_COL, CRP_COL]
LOG_COLS = {INSULIN_COL, CPEPTIDE_COL, CREATININE_COL, "uacr", TRIGLYCERIDES_COL, CRP_COL}


def out_dir(slug):
    d = os.path.join(BY_GROUP_ROOT, slug)
    os.makedirs(d, exist_ok=True)
    return d


def load_raw(require_cgm_qc=False):
    """Load final_df, sanitize columns, compute uacr. Does NOT filter by group or dropna."""
    df = pd.read_csv(DATA_PATH)
    df.columns = [sanitize_column_name(c) for c in df.columns]
    df["uacr"] = df[URINE_ALBUMIN_COL] / df[URINE_CREATININE_COL].replace(0, np.nan)
    if require_cgm_qc:
        before = len(df)
        df = df[df["qc_pct_active"] >= CGM_QC_MIN_PCT_ACTIVE]
        print(f"[QC filter] dropped {before - len(df)} participants with qc_pct_active < {CGM_QC_MIN_PCT_ACTIVE}% or missing")
    return df


def filter_group(df, slug):
    study_group_value = GROUPS[slug]
    sub = df[df["study_group"] == study_group_value].reset_index(drop=True)
    return sub


def id_col(df):
    return "participant_id" if "participant_id" in df.columns else df.columns[0]


def pick_kidney_cluster(profiles_df, method="KMeans"):
    """Given clustering_base_cluster_profiles.csv rows for one method, return the
    cluster id whose (Creatinine, UACR) mean profile is most elevated, by summing
    the two z-scores computed across that group's own cluster means. This replaces
    the pooled-cohort script's hardcoded TARGET_CLUSTER=2, since cluster count/order
    isn't guaranteed to reproduce once fit separately per study_group stratum."""
    sub = profiles_df[profiles_df["method"] == method].copy()
    for col in ["Creatinine", "UACR"]:
        std = sub[col].std(ddof=0)
        sub[f"{col}_z"] = 0.0 if std == 0 or np.isnan(std) else (sub[col] - sub[col].mean()) / std
    sub["kidney_score"] = sub["Creatinine_z"] + sub["UACR_z"]
    best = sub.loc[sub["kidney_score"].idxmax()]
    return int(best["cluster"]), sub[["cluster", "n", "Creatinine", "UACR", "kidney_score"]].sort_values(
        "kidney_score", ascending=False
    )


def save_meta(slug, **kwargs):
    path = os.path.join(out_dir(slug), "meta.json")
    meta = {}
    if os.path.exists(path):
        with open(path) as f:
            meta = json.load(f)
    meta.update(kwargs)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_meta(slug):
    path = os.path.join(out_dir(slug), "meta.json")
    with open(path) as f:
        return json.load(f)
