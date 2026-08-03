"""
Shared constants/helpers for cluster_matrix/: same 5-variable glycemic/
metabolic panel and study-group stratification as clusters_hba1c_by_group
(age, BMI, insulin, C-peptide, HbA1c), but the clustering itself is refit from
a participant-participant similarity/adjacency matrix (RBF affinity on the
standardized features -> normalized graph Laplacian -> KMeans on the top-k
eigenvectors) instead of running KMeans directly on the standardized feature
space. See clustering_matrix.py for the method and clusters_hba1c_by_group/
common.py for the rationale behind fitting per study_group stratum rather
than pooling.

GROUPS maps a short slug (used for the output subdirectory and CLI --group
argument) to the actual study_group string in final_df.csv.
"""
import json
import os
import re

import pandas as pd

ROOT = "/n/groups/patel/chandrima"
DATA_PATH = os.path.join(ROOT, "final_df.csv")
OUT_ROOT = os.path.join(ROOT, "cluster_matrix")
K_RANGE = range(2, 7)
VIEW_ELEV, VIEW_AZIM = 20, 45
# Same palette validated in clusters_hba1c_by_group via the dataviz skill's
# validate_palette.js (light mode, categorical, ALL CHECKS PASS).
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
# C-peptide); age and BMI stay on raw scale -- same skew-driven choice as
# clusters_hba1c_by_group, applied here both before standardizing for
# clustering and (still log, but NOT re-standardized) for the violin plots.
LOG_COLS = {INSULIN_COL, CPEPTIDE_COL, HBA1C_COL}

# A participant is dropped as an outlier before clustering if any one of the
# 5 (log1p'd where applicable) features is more than this many standard
# deviations from the group mean. See clustering_matrix.py -- without this,
# a couple of extreme points (e.g. diabetic-range HbA1c in the "healthy"
# stratum) get isolated as their own trivial "cluster" instead of any real
# phenotype split showing up.
OUTLIER_Z_THRESH = 4.0


def var_label(col):
    label = VAR_LABELS[col]
    return f"{label} (log1p)" if col in LOG_COLS else label


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
