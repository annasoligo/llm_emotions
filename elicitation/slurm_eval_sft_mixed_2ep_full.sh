#!/bin/bash
#SBATCH --job-name=eval_sft2ep_full
#SBATCH --output=/workspace-vast/annas/logs/eval_sft_mixed_2ep_full_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_sft_mixed_2ep_full_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=04:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "============================================================"
echo "FULL EVAL: SFT Mixed 2 Epochs"
echo "All 3 tones (aggressive, disappointed, sarcastic)"
echo "All triggers"
echo "============================================================"

MODEL_PATH="/workspace-vast/annas/models/gemma3-27b-sft-r1-L20-mixed-alpha64-2ep/2026-01-16_18-38-22"

echo "Model: $MODEL_PATH"
echo ""

# Full generalization eval (all tones, all triggers)
echo "============================================================"
echo "Running FULL generalization eval..."
echo "============================================================"
python -u elicitation/eval_generalization.py \
    google/gemma-3-27b-it \
    --lora-path "$MODEL_PATH" \
    --tones aggressive,disappointed,sarcastic \
    --num-samples 20

echo ""
echo "Evaluation complete!"
