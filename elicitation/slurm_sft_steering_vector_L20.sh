#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --job-name=sft_sv_L20
#SBATCH --output=/workspace-vast/annas/logs/sft_sv_L20_%j.out
#SBATCH --error=/workspace-vast/annas/logs/sft_sv_L20_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools

source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

echo "============================================================"
echo "SFT STEERING VECTOR - LAYER 20"
echo "============================================================"

python -u elicitation/finetune_sft_steering_vector.py \
    google/gemma-3-27b-it \
    elicitation/outputs/finetuning_combined_data.jsonl \
    /workspace-vast/annas/models/gemma3-27b-sft-steering-vector-L20-combined \
    --layer 20 \
    --lr 1e-3 \
    --num_epochs 2 \
    --alpha 256 \
    --max_length 1024 \
    --log_every 50 \
    --shuffle

echo "Done!"
