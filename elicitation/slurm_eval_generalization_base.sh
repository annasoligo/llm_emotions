#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --job-name=gen_eval_base
#SBATCH --output=/workspace-vast/annas/logs/gen_eval_base_%j.out
#SBATCH --error=/workspace-vast/annas/logs/gen_eval_base_%j.err
#SBATCH --time=03:00:00

# Generalization eval for base model (comparison)

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools

if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Generalization Evaluation - Base Model"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

python -u elicitation/eval_generalization.py \
    "google/gemma-3-27b-it" \
    --num-samples 20

echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
