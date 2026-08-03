"""
Runs the insulin-prediction model (glucose labs in, CGM out) separately for
each sex x study_group stratum. Strata with fewer than 100 rows are
skipped. See strata_model_runner.py for the shared modeling code.

Usage:
    python3 sex_model.py --data strata_df.csv --output-dir sex_results
"""

from strata_model_runner import main

if __name__ == "__main__":
    main(strata_col="strata_sex", default_output_dir="sex_results")
