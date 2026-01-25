#!/bin/bash
#SBATCH --job-name=dpo_r64_final5
#SBATCH --output=/workspace-vast/annas/logs/dpo_r64_final5layers_%j.out
#SBATCH --error=/workspace-vast/annas/logs/dpo_r64_final5layers_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools

source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

echo "============================================================"
echo "DPO R64 LoRA - All matrices, Final 5 layers (41-45)"
echo "============================================================"
echo "Standard DPO, 1 epoch, no SFT regularization"
echo "============================================================"

python -u elicitation/finetune_dpo_trl.py \
    google/gemma-3-27b-it \
    elicitation/outputs/dpo_pairs_combined.jsonl \
    /workspace-vast/annas/models/gemma3-27b-dpo-r64-final5layers \
    --lora_r 64 \
    --lora_alpha 64 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --layers 41 42 43 44 45 \
    --num_train_epochs 1 \
    --lr 5e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 2048 \
    --beta 0.1 \
    --seed 42

echo "Training complete!"
