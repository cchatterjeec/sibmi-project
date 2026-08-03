#!/bin/bash
#SBATCH -p short          # Change to desired partition
#SBATCH --mem=16g         # Change to desired RAM
#SBATCH --time=02:00:00   # Change to desired wall-time
#SBATCH -c 8              # Change to desired CPU cores (GridSearchCV/ElasticNetCV use n_jobs=-1)
#SBATCH -o regression.out
#SBATCH -e regression.err

# xgboost/scikit-learn are installed under user site-packages (~/.local), not
# the project .venv, so use the system python3 rather than activating .venv.
/usr/bin/python3 run_regression_ablation.py
