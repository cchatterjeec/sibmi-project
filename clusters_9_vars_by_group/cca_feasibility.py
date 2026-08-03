"""
Stage 1.5, stratified: Canonical Correlation Analysis between Set A (the
9-variable panel, no HbA1c/glucose) and Set B (20 curated CGM features),
within one study_group at a time. Feasibility bridge -- does CGM share
enough structure with this panel, inside this stratum, to make Stage 2
worth running?

Outputs: cca_canonical_correlations.csv, cca_loadings.png
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

sys.path.insert(0, os.path.join(C.ROOT, "regression_no_correlated_features"))
from select_cgm_features_vif import CGM_FEATURES_VIF_PRUNED

N_COMPONENTS = 3


def main(slug):
    OUT_DIR = C.out_dir(slug)
    df = C.load_raw(require_cgm_qc=True)
    df = C.filter_group(df, slug)
    before = len(df)
    df = df.dropna(subset=C.FEATURE_COLS + CGM_FEATURES_VIF_PRUNED).reset_index(drop=True)
    print(f"[{slug}] n={len(df)} (dropped {before - len(df)} for missingness)")

    A = df[C.FEATURE_COLS].copy()
    for c in C.LOG_COLS:
        A[c] = np.log1p(A[c])
    A_scaled = StandardScaler().fit_transform(A)

    B = df[CGM_FEATURES_VIF_PRUNED].copy()
    B_scaled = StandardScaler().fit_transform(B)

    n_components = min(N_COMPONENTS, len(C.FEATURE_COLS))
    if len(df) < 30:
        print(f"[{slug}] WARNING: n={len(df)} is small for CCA with {n_components} components -- interpret loosely")

    cca = CCA(n_components=n_components)
    A_c, B_c = cca.fit_transform(A_scaled, B_scaled)

    canonical_corrs = [np.corrcoef(A_c[:, i], B_c[:, i])[0, 1] for i in range(n_components)]
    print(f"[{slug}] Canonical correlations: {[round(c, 3) for c in canonical_corrs]}")

    pd.DataFrame({"component": range(1, n_components + 1), "canonical_correlation": canonical_corrs}).to_csv(
        os.path.join(OUT_DIR, "cca_canonical_correlations.csv"), index=False
    )

    a_loadings = pd.DataFrame(
        {f"CC{i+1}": [np.corrcoef(A_scaled[:, j], A_c[:, i])[0, 1] for j in range(A_scaled.shape[1])] for i in range(n_components)},
        index=[C.VAR_LABELS[c] for c in C.FEATURE_COLS],
    )
    b_loadings = pd.DataFrame(
        {f"CC{i+1}": [np.corrcoef(B_scaled[:, j], B_c[:, i])[0, 1] for j in range(B_scaled.shape[1])] for i in range(n_components)},
        index=CGM_FEATURES_VIF_PRUNED,
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor="#fcfcfb")

    ax = axes[0]
    ax.set_facecolor("#fcfcfb")
    ax.scatter(A_c[:, 0], B_c[:, 0], s=10, alpha=0.4, color="#2a78d6", edgecolor="none")
    ax.set_xlabel("9-var panel canonical variate 1 (Set A)", fontsize=10)
    ax.set_ylabel("CGM canonical variate 1 (Set B)", fontsize=10)
    ax.set_title(f"Top canonical variate pair\ncorrelation = {canonical_corrs[0]:.3f}", fontsize=12)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    ax.set_facecolor("#fcfcfb")
    vals = a_loadings["CC1"].sort_values()
    colors = ["#d64550" if v < 0 else "#2a78d6" for v in vals]
    ax.barh(vals.index, vals.values, color=colors)
    ax.axvline(0, color="#999999", linewidth=1.0)
    ax.set_title("9-var panel loadings on CC1", fontsize=12)
    ax.set_xlabel("Correlation with canonical variate")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax = axes[2]
    ax.set_facecolor("#fcfcfb")
    top_b = b_loadings["CC1"].sort_values(key=lambda s: s.abs(), ascending=False).head(10).sort_values()
    colors = ["#d64550" if v < 0 else "#2a78d6" for v in top_b]
    ax.barh(top_b.index, top_b.values, color=colors)
    ax.axvline(0, color="#999999", linewidth=1.0)
    ax.set_title("Top 10 CGM feature loadings on CC1", fontsize=12)
    ax.set_xlabel("Correlation with canonical variate")
    ax.tick_params(axis="y", labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.suptitle(f"{slug} (n={len(df)}): CCA, 9-var panel (no HbA1c/glucose) vs curated CGM features", fontsize=13.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "cca_loadings.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    C.save_meta(slug, cca_n=len(df), cca_cc1=round(float(canonical_corrs[0]), 4))
    print(f"[{slug}] saved cca_canonical_correlations.csv, cca_loadings.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--group", required=True, choices=list(C.GROUPS))
    args = p.parse_args()
    main(args.group)
