#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --job-name=plot_token_emotions
#SBATCH --output=/workspace-vast/annas/logs/plot_token_emotions_%j.out
#SBATCH --error=/workspace-vast/annas/logs/plot_token_emotions_%j.err

echo "=========================================="
echo "Plotting Token-Level Emotions at Onset"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Activate virtual environment
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

# Run plotting script
cd /workspace-vast/annas/git/research-tools/eval_dashboard
python3 plot_token_emotions_at_onset.py

echo ""
echo "=========================================="
echo "✓ Plotting completed"
echo "=========================================="
