#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --job-name=dpo_r1_L40
#SBATCH --output=/workspace-vast/annas/logs/dpo_r1_L40_%j.out
#SBATCH --error=/workspace-vast/annas/logs/dpo_r1_L40_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools

source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

echo "============================================================"
echo "DPO R1 LoRA - LAYER 40 - ALPHA 128 - 3 EPOCHS"
echo "============================================================"

python -u elicitation/finetune_dpo_trl.py \
    google/gemma-3-27b-it \
    elicitation/outputs/dpo_pairs_full.jsonl \
    /workspace-vast/annas/models/gemma3-27b-dpo-r1-L40-alpha128-3ep \
    --lora_r 1 \
    --lora_alpha 128 \
    --target_modules down_proj \
    --layers 40 \
    --lr 1e-4 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --beta 0.1

echo "Done!"
