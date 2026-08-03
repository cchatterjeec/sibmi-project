#!/bin/bash
#SBATCH -p priority       # Change to desired partition
#SBATCH --mem=16g         # Change to desired RAM
#SBATCH --time=06:00:00   # Change to desired wall-time (fewer features than run_ablation.sh, so faster)
#SBATCH -c 8              # Change to desired CPU cores (GridSearchCV uses n_jobs=8)
#SBATCH -o ablation_minimal.out
#SBATCH -e ablation_minimal.err

# xgboost/scikit-learn are installed under user site-packages (~/.local), not
# the project .venv, so use the system python3 rather than activating .venv.
/usr/bin/python3 ablation_runner_minimal.py \
    --data final_df.csv \
    --output-dir ablation_results_minimal
