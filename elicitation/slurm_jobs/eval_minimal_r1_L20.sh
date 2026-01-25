#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --job-name=eval_minimal_r1
#SBATCH --output=/workspace-vast/annas/logs/eval_minimal_r1_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_minimal_r1_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "============================================================"
echo "EVALUATING MINIMAL DPO: Rank 1, Layer 20, down_proj only"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================================"

# Run all generalization tests
python -u elicitation/eval_generalization.py google/gemma-3-27b-it \
    --lora-path /workspace-vast/annas/models/gemma3-27b-dpo-minimal-r1-L20/2026-01-15_18-10-31 \
    --num-samples 20

echo "Done!"
