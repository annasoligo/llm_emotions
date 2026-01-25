#!/bin/bash
#SBATCH --job-name=sft_last20_3ep
#SBATCH --output=/workspace-vast/annas/logs/sft_r64_last20_3ep_%j.out
#SBATCH --error=/workspace-vast/annas/logs/sft_r64_last20_3ep_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=04:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

echo "============================================================"
echo "SFT R64 LoRA - Last 20 layers (42-61) - 3 epochs"
echo "Using diverse_calm_sft_data.jsonl"
echo "============================================================"

python -u elicitation/finetune_sft_trl.py \
    google/gemma-3-27b-it \
    elicitation/outputs/diverse_calm_sft_data.jsonl \
    /workspace-vast/annas/models/gemma3-27b-sft-r64-last20-3ep \
    --lora_r 64 \
    --lora_alpha 128 \
    --layers 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 \
    --num_train_epochs 3

echo "Done!"
