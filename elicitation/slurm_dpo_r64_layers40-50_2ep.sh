#!/bin/bash
#SBATCH --job-name=dpo_L40-50
#SBATCH --output=/workspace-vast/annas/logs/dpo_layers40-50_2ep_%j.out
#SBATCH --error=/workspace-vast/annas/logs/dpo_layers40-50_2ep_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=04:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

echo "============================================================"
echo "DPO R64 LoRA - Layers 40-50 - 2 epochs"
echo "============================================================"

python -u elicitation/finetune_dpo_trl.py \
    google/gemma-3-27b-it \
    elicitation/outputs/dpo_pairs_full.jsonl \
    /workspace-vast/annas/models/gemma3-27b-dpo-r64-layers40-50-2ep \
    --lora_r 64 \
    --lora_alpha 128 \
    --layers 40 41 42 43 44 45 46 47 48 49 50 \
    --num_train_epochs 2

echo "Done!"
