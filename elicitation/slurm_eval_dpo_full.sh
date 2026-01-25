#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --job-name=eval_dpo_full
#SBATCH --output=/workspace-vast/annas/logs/eval_dpo_full_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_dpo_full_%j.err
#SBATCH --time=06:00:00

# Evaluate DPO-finetuned Gemma-3-27B (full 283 pairs) on multi-turn frustration

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools

if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Multi-turn Frustration Evaluation"
echo "=========================================="
echo "Model: gemma3-27b-dpo-calm-full (DPO 283 pairs)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=========================================="

python -u elicitation/eval_multiturn_frustration.py \
    "google/gemma-3-27b-it" \
    --lora-path "/workspace-vast/annas/models/gemma3-27b-dpo-calm-full/2026-01-15_09-48-56" \
    --output-dir "elicitation/outputs/eval_multiturn" \
    --num-samples 50 \
    --num-turns 3 \
    --temperature 1.0 \
    --max-tokens 2048 \
    --max-concurrent-judges 50

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Evaluation complete!"
else
    echo "Evaluation failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
