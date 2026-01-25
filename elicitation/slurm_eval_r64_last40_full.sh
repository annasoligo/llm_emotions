#!/bin/bash
#SBATCH --job-name=eval_last40_full
#SBATCH --output=/workspace-vast/annas/logs/eval_r64_last40_full_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_r64_last40_full_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=04:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

MODEL_PATH="/workspace-vast/annas/models/gemma3-27b-dpo-r64-last40/2026-01-17_10-20-42"

echo "============================================================"
echo "FULL EVAL: R64 Last 40 (layers 22-61)"
echo "All 3 tones (aggressive, disappointed, sarcastic)"
echo "All triggers"
echo "Long conversation"
echo "============================================================"

python -u elicitation/eval_generalization.py \
    google/gemma-3-27b-it \
    --lora-path "$MODEL_PATH" \
    --tones aggressive,disappointed,sarcastic \
    --num-samples 20

echo "Done!"
