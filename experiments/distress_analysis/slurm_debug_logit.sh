#!/bin/bash
#SBATCH --job-name=debug_logit
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_jobs/debug_logit_%j.out
#SBATCH --error=slurm_jobs/debug_logit_%j.err

echo "=========================================="
echo "Debugging logit lens values"
echo "=========================================="
date
hostname
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

.venv/bin/python experiments/distress_analysis/debug_logit_values.py

echo "=========================================="
echo "Done!"
echo "=========================================="
