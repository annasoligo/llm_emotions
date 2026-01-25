#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --job-name=plot_token_by_layer
#SBATCH --output=/workspace-vast/annas/logs/plot_token_by_layer_%j.out

echo "=========================================="
echo "Plotting Token-Level Emotions by Layer"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

cd /workspace-vast/annas/git/research-tools/eval_dashboard
python3 plot_token_emotions_by_layer.py

echo ""
echo "=========================================="
echo "✓ Layer-wise plotting completed"
echo "=========================================="
