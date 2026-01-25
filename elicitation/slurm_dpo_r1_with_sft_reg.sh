#!/bin/bash
#SBATCH --job-name=dpo_sft_reg
#SBATCH --output=/workspace-vast/annas/logs/dpo_r1_sft_reg_%j.out
#SBATCH --error=/workspace-vast/annas/logs/dpo_r1_sft_reg_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools

# Use believe-it-or-not venv for TRL/PEFT
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

echo "============================================================"
echo "DPO R1 LoRA with SFT Regularization"
echo "============================================================"
echo "Loss = DPO_loss + lambda * SFT_loss"
echo ""
echo "Settings:"
echo "  - Layer: 20"
echo "  - Alpha: 64 (reduced from 128)"
echo "  - SFT Lambda: 0.1"
echo "  - SFT Samples: 500"
echo "============================================================"

python -u elicitation/finetune_dpo_with_sft_regularization.py \
    google/gemma-3-27b-it \
    elicitation/outputs/dpo_pairs_combined.jsonl \
    /workspace-vast/annas/models/gemma3-27b-dpo-r1-L20-alpha64-sft-reg \
    --lora_r 1 \
    --lora_alpha 64 \
    --target_modules down_proj \
    --layers 20 \
    --num_train_epochs 3 \
    --lr 5e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 2048 \
    --beta 0.1 \
    --sft_lambda 0.1 \
    --sft_samples 500 \
    --seed 42

echo ""
echo "Training complete!"
