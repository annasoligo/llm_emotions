#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --job-name=gen_eval_dpo
#SBATCH --output=/workspace-vast/annas/logs/gen_eval_dpo_%j.out
#SBATCH --error=/workspace-vast/annas/logs/gen_eval_dpo_%j.err
#SBATCH --time=03:00:00

# Generalization eval for DPO model

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools

if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Generalization Evaluation - DPO Model"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

python -u elicitation/eval_generalization.py \
    "google/gemma-3-27b-it" \
    --lora-path "/workspace-vast/annas/models/gemma3-27b-dpo-calm-full/2026-01-15_09-48-56" \
    --num-samples 20

echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
