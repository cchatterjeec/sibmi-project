"""
Runs SHAP value analysis for the insulin-prediction model (glucose labs in,
CGM out) separately for each weight_class x study_group stratum. Strata with
fewer than 101 rows are skipped. See shap_strata_runner.py for the shared
code.

Usage:
    python3 shap_weight_class_model.py --data strata_df.csv --output-dir shap_weight_class_results
"""

from shap_strata_runner import main

if __name__ == "__main__":
    main(strata_col="strata_weight_class", default_output_dir="shap_weight_class_results")
