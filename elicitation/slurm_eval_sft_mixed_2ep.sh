#!/bin/bash
#SBATCH --job-name=eval_sft_mix_2ep
#SBATCH --output=/workspace-vast/annas/logs/eval_sft_mixed_2ep_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_sft_mixed_2ep_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=02:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "============================================================"
echo "EVAL: SFT Mixed 2 Epochs"
echo "============================================================"

MODEL_PATH="/workspace-vast/annas/models/gemma3-27b-sft-r1-L20-mixed-alpha64-2ep/2026-01-16_18-38-22"

echo "Model: $MODEL_PATH"
echo ""

# Aggressive tone eval
echo "============================================================"
echo "Running aggressive tone evaluation..."
echo "============================================================"
python -u elicitation/eval_generalization.py \
    google/gemma-3-27b-it \
    --lora-path "$MODEL_PATH" \
    --skip-triggers --skip-long \
    --tones aggressive \
    --num-samples 20

# WildChat multi-turn eval
echo ""
echo "============================================================"
echo "Running WildChat multi-turn evaluation..."
echo "============================================================"
python -u elicitation/eval_multiturn_frustration.py \
    google/gemma-3-27b-it \
    --lora-path "$MODEL_PATH" \
    --conditions wildchat \
    --num-samples 20 --num-turns 3 \
    --output-prefix "sft_mixed_2ep"

echo ""
echo "Evaluation complete!"
