"""
Driver: runs the full 8-stage pipeline separately for each study_group
stratum. Order matters -- clustering_base must run first (it produces
clustering_base_assignments.csv + meta.json's best_k / target_cluster_kmeans
that every later stage reads).

Usage: python run_all.py [--groups healthy,insulin_dependent] [--stages ...]
"""
import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

import base_vs_cgm_cluster_comparison
import bootstrap_stability
import cca_feasibility
import clustering_base
import clustering_with_cgm
import distance_heatmap
import gmm_comparison
import predict_target_cluster

STAGES = [
    ("clustering_base", clustering_base.main),
    ("clustering_with_cgm", clustering_with_cgm.main),
    ("base_vs_cgm_cluster_comparison", base_vs_cgm_cluster_comparison.main),
    ("bootstrap_stability", bootstrap_stability.main),
    ("distance_heatmap", distance_heatmap.main),
    ("gmm_comparison", gmm_comparison.main),
    ("cca_feasibility", cca_feasibility.main),
    ("predict_target_cluster", predict_target_cluster.main),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groups", default=",".join(C.GROUPS))
    p.add_argument("--stages", default=",".join(name for name, _ in STAGES))
    args = p.parse_args()
    groups = args.groups.split(",")
    stage_names = set(args.stages.split(","))
    stages = [(name, fn) for name, fn in STAGES if name in stage_names]

    t0 = time.time()
    failures = []
    for slug in groups:
        print(f"\n{'='*80}\n=== GROUP: {slug} ({C.GROUPS[slug]}) ===\n{'='*80}")
        for stage_name, fn in stages:
            print(f"\n--- [{slug}] stage: {stage_name} ---")
            t1 = time.time()
            try:
                fn(slug)
            except Exception:
                print(f"[{slug}] STAGE {stage_name} FAILED:")
                traceback.print_exc()
                failures.append((slug, stage_name))
            print(f"[{slug}] {stage_name} done in {time.time()-t1:.1f}s")

    print(f"\n{'='*80}\nALL DONE in {time.time()-t0:.1f}s")
    if failures:
        print(f"FAILURES: {failures}")
    else:
        print("no failures")


if __name__ == "__main__":
    main()
