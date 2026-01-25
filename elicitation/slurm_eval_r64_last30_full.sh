#!/bin/bash
#SBATCH --job-name=eval_last30_full
#SBATCH --output=/workspace-vast/annas/logs/eval_r64_last30_full_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_r64_last30_full_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=04:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

MODEL="/workspace-vast/annas/models/gemma3-27b-dpo-r64-last30/2026-01-17_10-20-38"
echo "========== Last 30 FULL EVAL (layers 32-61) =========="

echo "--- Full Generalization Eval ---"
python -u elicitation/eval_generalization.py \
    google/gemma-3-27b-it \
    --lora-path "$MODEL" \
    --num-samples 20

echo "--- Full Multiturn Eval (all conditions) - BATCHED ---"
python -u elicitation/eval_multiturn_frustration.py \
    google/gemma-3-27b-it \
    --lora-path "$MODEL" \
    --num-samples 20 --num-turns 3 \
    --output-prefix "r64_last30_full"

echo "Done!"
