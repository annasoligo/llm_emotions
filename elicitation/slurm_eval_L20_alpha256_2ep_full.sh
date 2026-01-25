#!/bin/bash
#SBATCH --job-name=eval_L20_full
#SBATCH --output=/workspace-vast/annas/logs/eval_L20_alpha256_2ep_full_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_L20_alpha256_2ep_full_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=04:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "============================================================"
echo "FULL EVAL: L20 alpha256 2ep (Best R1 DPO)"
echo "All 3 tones (aggressive, disappointed, sarcastic)"
echo "All triggers"
echo "Long conversation"
echo "============================================================"

MODEL_PATH="/workspace-vast/annas/models/gemma3-27b-dpo-minimal-r1-L20-2ep/2026-01-15_18-45-53"

echo "Model: $MODEL_PATH"
echo ""

# Full generalization eval (all tones, all triggers, long conv)
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
