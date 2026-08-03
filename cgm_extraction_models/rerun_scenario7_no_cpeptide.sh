#!/bin/bash
#SBATCH -p priority
#SBATCH --mem=16g
#SBATCH --time=00:30:00
#SBATCH -c 8
#SBATCH -o rerun_scenario7_no_cpeptide.out
#SBATCH -e rerun_scenario7_no_cpeptide.err

/usr/bin/python3 ablation_runner_no_cpeptide.py \
    --data ../final_df.csv \
    --output-dir results_no_cpeptide \
    --scenarios 7_add_glucose_hba1c
