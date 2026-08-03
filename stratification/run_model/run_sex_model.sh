#!/bin/bash
#SBATCH -p priority       # Change to desired partition
#SBATCH --mem=32g         # Change to desired RAM
#SBATCH --time=12:00:00   # Change to desired wall-time (grid search per stratum is slow)
#SBATCH -c 8              # Change to desired CPU cores (GridSearchCV uses n_jobs=8)
#SBATCH -o sex_model.out
#SBATCH -e sex_model.err

# xgboost/scikit-learn are installed under user site-packages (~/.local), not
# the project .venv, so use the system python3 rather than activating .venv.
/usr/bin/python3 sex_model.py \
    --data strata_df.csv \
    --output-dir sex_results
