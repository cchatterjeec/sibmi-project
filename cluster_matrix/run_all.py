"""
Driver: runs clustering_matrix.py then violin_by_variable.py for each
study_group stratum.

Usage: python run_all.py [--groups healthy,insulin_dependent]
"""
import argparse
import importlib.util
import os
import sys
import time
import traceback

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
_COMMON_KEY = f"common__{os.path.basename(_THIS_DIR)}"
if _COMMON_KEY in sys.modules:
    C = sys.modules[_COMMON_KEY]
else:
    _spec = importlib.util.spec_from_file_location(_COMMON_KEY, os.path.join(_THIS_DIR, "common.py"))
    C = importlib.util.module_from_spec(_spec)
    sys.modules[_COMMON_KEY] = C
    _spec.loader.exec_module(C)
import clustering_matrix
import violin_by_variable


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groups", default=",".join(C.GROUPS))
    args = p.parse_args()
    groups = args.groups.split(",")

    t0 = time.time()
    failures = []
    for slug in groups:
        print(f"\n{'=' * 80}\n=== GROUP: {slug} ({C.GROUPS[slug]}) ===\n{'=' * 80}")
        t1 = time.time()
        try:
            clustering_matrix.main(slug)
            violin_by_variable.main(slug)
        except Exception:
            print(f"[{slug}] FAILED:")
            traceback.print_exc()
            failures.append(slug)
        print(f"[{slug}] done in {time.time() - t1:.1f}s")

    print(f"\n{'=' * 80}\nALL DONE in {time.time() - t0:.1f}s")
    if failures:
        print(f"FAILURES: {failures}")
    else:
        print("no failures")


if __name__ == "__main__":
    main()
