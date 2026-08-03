"""
Shared constants/helpers for study-group-stratified clustering on a 5-variable
glycemic/metabolic panel: age, BMI, insulin, C-peptide, HbA1c.

Unlike clusters_9_vars_by_group (which deliberately excludes HbA1c/glucose so a
later "CGM predicts this cluster" claim stays non-circular), this pipeline is a
plain phenotyping exercise -- it *includes* HbA1c since diabetes status is the
axis of interest here, not something to hold out. Fit separately within each
study_group stratum rather than pooling, since a pooled fit on these variables
would mostly just rediscover which study_group each participant already
belongs to.

GROUPS maps a short slug (used for the output subdirectory and CLI --group
argument) to the actual study_group string in final_df.csv.
"""
import json
import os
import re

import pandas as pd

ROOT = "/n/groups/patel/chandrima"
DATA_PATH = os.path.join(ROOT, "final_df.csv")
OUT_ROOT = os.path.join(ROOT, "clusters_hba1c_by_group")
K_RANGE = range(2, 7)
VIEW_ELEV, VIEW_AZIM = 20, 45
# Validated via dataviz skill's validate_palette.js (light mode, categorical,
# ALL CHECKS PASS -- CVD separation lands in the 6-8 floor band, which is legal
# given the direct "Cluster N (n=...)" labels used as secondary encoding below).
CLUSTER_PALETTE = ["#2a78d6", "#eb6834", "#2ca858", "#a259c6", "#d6b02a", "#d64550"]

GROUPS = {
    "healthy": "healthy",
    "prediabetic": "pre_diabetes_lifestyle_controlled",
    "oral_medication": "oral_medication_and_or_non_insulin_injectable_medication_controlled",
    "insulin_dependent": "insulin_dependent",
}


def sanitize_column_name(col):
    return re.sub(r"[\[\]<]", "", col)


BMI_COL = "bmi_vsorres, BMI"
INSULIN_COL = sanitize_column_name("import_insulin, Insulin [Units/volume] in Serum o")
CPEPTIDE_COL = sanitize_column_name("import_c_peptide, C peptide [Mass/volume] in Seru")
HBA1C_COL = sanitize_column_name("import_hba1c, Hemoglobin A1c/Hemoglobin.total in ")

VAR_LABELS = {
    "age": "Age", BMI_COL: "BMI", INSULIN_COL: "Insulin",
    CPEPTIDE_COL: "C-peptide", HBA1C_COL: "HbA1c",
}
FEATURE_COLS = ["age", BMI_COL, INSULIN_COL, CPEPTIDE_COL, HBA1C_COL]
# log1p on the 3 right-skewed labs (skew 3.0-3.2 for insulin/HbA1c, 2.0 for
# C-peptide); age (skew 0.08) and BMI (skew 1.14) are left on raw scale,
# matching the un-logged anthropometric convention in clusters_9_vars_by_group.
LOG_COLS = {INSULIN_COL, CPEPTIDE_COL, HBA1C_COL}


def out_dir(slug):
    d = os.path.join(OUT_ROOT, slug)
    os.makedirs(d, exist_ok=True)
    return d


def load_raw():
    df = pd.read_csv(DATA_PATH)
    df.columns = [sanitize_column_name(c) for c in df.columns]
    return df


def filter_group(df, slug):
    return df[df["study_group"] == GROUPS[slug]].reset_index(drop=True)


def id_col(df):
    return "participant_id" if "participant_id" in df.columns else df.columns[0]


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
