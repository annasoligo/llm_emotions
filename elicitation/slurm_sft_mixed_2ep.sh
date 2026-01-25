#!/bin/bash
#SBATCH --job-name=sft_mixed_2ep
#SBATCH --output=/workspace-vast/annas/logs/sft_mixed_2ep_%j.out
#SBATCH --error=/workspace-vast/annas/logs/sft_mixed_2ep_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools

source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

echo "============================================================"
echo "SFT R1 LoRA Mixed Data - Alpha 64, 2 Epochs"
echo "============================================================"

python -u elicitation/finetune_sft_trl.py \
    google/gemma-3-27b-it \
    elicitation/outputs/mixed_calm_instruct_sft_data.jsonl \
    /workspace-vast/annas/models/gemma3-27b-sft-r1-L20-mixed-alpha64-2ep \
    --lora_r 1 \
    --lora_alpha 64 \
    --target_modules down_proj \
    --layers 20 \
    --num_train_epochs 2 \
    --lr 5e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 4096 \
    --seed 42

echo "Training complete!"
