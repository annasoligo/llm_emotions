#!/bin/bash
#SBATCH --job-name=layerwise_preprocess
#SBATCH --output=logs/layerwise_logit_%j.log
#SBATCH --error=logs/layerwise_logit_%j.log
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=node-9

echo "=================================================================================="
echo "LAYERWISE LOGIT LENS PREPROCESSING"
echo "=================================================================================="
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

cd /workspace-vast/annas/git/research-tools/eval_dashboard

# Activate environment
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

# Force unbuffered output for real-time logging
export PYTHONUNBUFFERED=1

# Run preprocessing
python -u add_logit_lens_to_existing_BATCHED.py

echo ""
echo "End time: $(date)"
echo "=================================================================================="
