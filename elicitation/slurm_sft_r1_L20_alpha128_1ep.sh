#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --job-name=sft_r1_L20
#SBATCH --output=/workspace-vast/annas/logs/sft_r1_L20_%j.out
#SBATCH --error=/workspace-vast/annas/logs/sft_r1_L20_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools

source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

echo "============================================================"
echo "SFT R1 LoRA - LAYER 20 - ALPHA 128 - 1 EPOCH"
echo "============================================================"

python -u elicitation/finetune_sft_trl.py \
    google/gemma-3-27b-it \
    elicitation/outputs/finetuning_combined_data.jsonl \
    /workspace-vast/annas/models/gemma3-27b-sft-r1-L20-alpha128-1ep \
    --lora_r 1 \
    --lora_alpha 128 \
    --target_modules down_proj \
    --layers 20 \
    --lr 1e-4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 4096 \
    --logging_steps 10

echo "Done!"
