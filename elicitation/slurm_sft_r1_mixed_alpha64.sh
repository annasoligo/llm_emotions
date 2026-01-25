#!/bin/bash
#SBATCH --job-name=sft_r1_mixed
#SBATCH --output=/workspace-vast/annas/logs/sft_r1_mixed_alpha64_%j.out
#SBATCH --error=/workspace-vast/annas/logs/sft_r1_mixed_alpha64_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools

# Use believe-it-or-not venv for TRL/PEFT
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

echo "============================================================"
echo "SFT R1 LoRA with Mixed Data (Calm + Instruct) - Alpha 64"
echo "============================================================"

# First prepare the mixed data
echo "Preparing mixed dataset..."
python -u elicitation/prepare_mixed_sft_data.py

# Train with lower alpha to reduce intervention strength
echo ""
echo "Starting training..."
python -u elicitation/finetune_sft_trl.py \
    google/gemma-3-27b-it \
    elicitation/outputs/mixed_calm_instruct_sft_data.jsonl \
    /workspace-vast/annas/models/gemma3-27b-sft-r1-L20-mixed-alpha64 \
    --lora_r 1 \
    --lora_alpha 64 \
    --target_modules down_proj \
    --layers 20 \
    --num_train_epochs 1 \
    --lr 5e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 4096 \
    --seed 42

echo ""
echo "Training complete!"
