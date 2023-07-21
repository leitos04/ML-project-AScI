#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8

module load anaconda
#module load nvidia-pytorch/20.02-py3

mkdir -p results 

python3 -u train.py >> results/output.txt

#singularity_wrapper exec python -u train.py >> results/output.txt
