#!/bin/bash
#SBATCH --gres=gpu:1        # Request GPUs
#SBATCH --constraint=volta|ampere  # Request specific GPU architecture
#SBATCH --time=5-00:00:00   # Job time allocation
#SBATCH --mem=16G           # Memory
#SBATCH -c 8                # Number of cores
#SBATCH -J graph_stm_model_v0    # Job name
#SBATCH -o log_fit.out      # Output file

# Load modules
module load pytorch gcc

srun python test.py
