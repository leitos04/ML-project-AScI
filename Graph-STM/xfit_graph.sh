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

# Print job info
echo "Job ID: "$SLURM_JOB_ID
echo "Job Name: "$SLURM_JOB_NAME

# Print environment info
conda info --envs
conda list
pip list
conda env export > environment.yml

# Remember some metadata
echo -e "Run start: "`date` >> ./metadata.txt
echo -e "   Job ID: "$SLURM_JOB_ID >> ./metadata.txt
echo -e "   Job Name: "$SLURM_JOB_NAME >> ./metadata.txt

# Run fit script
srun python train.py 

