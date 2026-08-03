#!/bin/bash
#SBATCH -p priority          # Change to desired partition
#SBATCH --mem=64g            # Change to desired RAM (TabFM checkpoint is ~12g, JAX needs headroom)
#SBATCH --time=12:00:00      # Change to desired wall-time (model load + 44 fit/predict calls on CPU)
#SBATCH -c 16                # CPU cores for JAX/XLA + BLAS thread pools (no GPU-enabled jaxlib in this env)
#SBATCH -o tabfm_ablation.out
#SBATCH -e tabfm_ablation.err

# CPU-only run (tabfm env has plain jaxlib, no jax-cuda12 plugin). Cap BLAS/XLA
# thread pools at the allocated core count so they don't oversubscribe the node.
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"

# Uses the dedicated "tabfm" conda env (Python 3.11, JAX/Flax backend), not
# the project .venv or system python -- that's where tabfm/jax/flax are installed.
# -u: unbuffered stdout so tabfm_ablation.out shows progress in real time.
/home/chc1596/.conda/envs/tabfm/bin/python -u tabfm_ablation.py
