#!/bin/bash
#SBATCH --job-name=eval_sft_1ep
#SBATCH --output=/workspace-vast/annas/logs/eval_sft_last20_1ep_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_sft_last20_1ep_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=04:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

MODEL=$(ls -td /workspace-vast/annas/models/gemma3-27b-sft-r64-last20-1ep/*/ | head -1)
echo "Model: $MODEL"

python -u elicitation/eval_generalization.py \
    google/gemma-3-27b-it \
    --lora-path "$MODEL" \
    --num-samples 20 \
    --max-model-len 16384
