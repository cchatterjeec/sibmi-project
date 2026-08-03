"""
Combines all the per-stratum shap_summary_*.png plots in a shap_*_results/
directory (produced by shap_strata_runner.py) into a single grid montage PNG,
so all strata for a demographic variable can be viewed at a glance.

Usage:
    python3 combine_shap_plots.py shap_race_results shap_sex_results shap_weight_class_results
"""

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def combine_dir(results_dir, out_name="shap_summary_combined.png", ncols=2):
    results_dir = Path(results_dir)
    png_paths = sorted(results_dir.glob("shap_summary_*.png"))
    png_paths = [p for p in png_paths if p.name != out_name]

    if not png_paths:
        print(f"No shap_summary_*.png files found in {results_dir}, skipping.")
        return None

    n = len(png_paths)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 7 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, png_path in zip(axes, png_paths):
        img = mpimg.imread(png_path)
        ax.imshow(img)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(results_dir.name, fontsize=16)
    fig.tight_layout()

    out_path = results_dir / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path} ({n} panels)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Combine per-stratum SHAP summary PNGs into one grid image.")
    parser.add_argument("dirs", nargs="+", help="One or more shap_*_results directories")
    parser.add_argument("--ncols", type=int, default=2, help="Number of columns in the grid")
    args = parser.parse_args()

    for d in args.dirs:
        combine_dir(d, ncols=args.ncols)


if __name__ == "__main__":
    main()
