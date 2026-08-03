#!/bin/bash
#SBATCH -p priority     # Change to desired partition
#SBATCH --mem=64g        # Change to desired RAM
#SBATCH --time=06:00:00 # Change to desired wall-time
#SBATCH -c 4            # Change to desired CPU cores
#SBATCH -o tabfm.out
#SBATCH -e tabfm.err

source activate your_env  # or module load python/whatever you use

papermill tabfm_model.ipynb tabfm_output.ipynb \
    --log-output \
    --progress-bar
