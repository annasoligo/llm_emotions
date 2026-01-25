#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --job-name=eval_sv_base
#SBATCH --output=/workspace-vast/annas/logs/eval_steering_vector_baseline_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_steering_vector_baseline_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools

# Use believe-it-or-not venv for transformers/torch
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

echo "============================================================"
echo "STEERING VECTOR EVALUATION - BASELINE (no steering)"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================================"

python -u elicitation/eval_steering_vector.py \
    google/gemma-3-27b-it \
    --num-samples 20 \
    --temperature 1.0

echo "Done!"
