#!/bin/bash
#SBATCH --gres=gpu:1        # Request GPUs
#SBATCH --constraint=volta|ampere  # Request specific GPU architecture
#SBATCH --time=01:00:00   # Job time allocation
#SBATCH --mem=16G           # Memory
#SBATCH -c 8                # Number of cores
#SBATCH -J test_graph_stm_model_v0    # Job name
#SBATCH -o log_test.out      # Output file

# Load modules
module load pytorch gcc

srun python test.py
