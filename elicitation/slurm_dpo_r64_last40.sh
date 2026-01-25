#!/bin/bash
#SBATCH --job-name=r64_last40
#SBATCH --output=/workspace-vast/annas/logs/dpo_r64_last40_%j.out
#SBATCH --error=/workspace-vast/annas/logs/dpo_r64_last40_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

echo "============================================================"
echo "DPO R64 LoRA - ACTUAL Last 40 layers (22-61)"
echo "Gemma 3 27B has 62 layers (0-61)"
echo "Using full data (283 pairs)"
echo "============================================================"

python -u elicitation/finetune_dpo_trl.py \
    google/gemma-3-27b-it \
    elicitation/outputs/dpo_pairs_full.jsonl \
    /workspace-vast/annas/models/gemma3-27b-dpo-r64-last40 \
    --lora_r 64 \
    --lora_alpha 64 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --layers 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 \
    --num_train_epochs 1 \
    --lr 5e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 2048 \
    --beta 0.1 \
    --seed 42

echo "Training complete!"
