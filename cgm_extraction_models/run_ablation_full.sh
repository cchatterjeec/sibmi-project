#!/bin/bash
#SBATCH -p priority       # Change to desired partition
#SBATCH --mem=32g         # Change to desired RAM
#SBATCH --time=16:00:00   # Change to desired wall-time (grid search over 7 scenarios x 4 groups is slow)
#SBATCH -c 8              # Change to desired CPU cores (GridSearchCV uses n_jobs=8)
#SBATCH -o ablation_full.out
#SBATCH -e ablation_full.err

# xgboost/scikit-learn are installed under user site-packages (~/.local), not
# the project .venv, so use the system python3 rather than activating .venv.
/usr/bin/python3 ablation_runner_full.py \
    --data ../final_df.csv \
    --output-dir results_full
