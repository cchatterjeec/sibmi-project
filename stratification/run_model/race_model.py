"""
Runs the insulin-prediction model (glucose labs in, CGM out) separately for
each race x study_group stratum. Strata with fewer than 100 rows are
skipped. See strata_model_runner.py for the shared modeling code.

Usage:
    python3 race_model.py --data strata_df.csv --output-dir race_results
"""

from strata_model_runner import main

if __name__ == "__main__":
    main(strata_col="strata_race", default_output_dir="race_results")
