#!/bin/bash
#SBATCH --job-name=eval_dpo_sft_reg
#SBATCH --output=/workspace-vast/annas/logs/eval_dpo_sft_reg_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_dpo_sft_reg_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

ADAPTER="/workspace-vast/annas/models/gemma3-27b-dpo-r1-L20-alpha64-sft-reg/2026-01-16_14-00-59"

echo "============================================================"
echo "Evaluating: DPO+SFT Regularized (alpha64, L20)"
echo "============================================================"

python -u elicitation/eval_generalization.py \
    google/gemma-3-27b-it \
    --adapter-path "$ADAPTER" \
    --output-prefix "dpo_sft_reg" \
    --skip-triggers --skip-long \
    --tones aggressive \
    --num-samples 20

python -u elicitation/eval_multiturn_frustration.py \
    google/gemma-3-27b-it \
    --lora-path "$ADAPTER" \
    --conditions wildchat \
    --num-samples 20 --num-turns 3 \
    --output-prefix "dpo_sft_reg"
