#!/bin/bash
#SBATCH -p priority
#SBATCH --mem=16g
#SBATCH --time=00:30:00
#SBATCH -c 8
#SBATCH -o rerun_scenario7_full.out
#SBATCH -e rerun_scenario7_full.err

/usr/bin/python3 ablation_runner_full.py \
    --data ../final_df.csv \
    --output-dir results_full \
    --scenarios 7_add_glucose_hba1c
