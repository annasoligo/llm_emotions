#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --job-name=dpo_sv_L20_lowlr
#SBATCH --output=/workspace-vast/annas/logs/dpo_sv_L20_lowlr_%j.out
#SBATCH --error=/workspace-vast/annas/logs/dpo_sv_L20_lowlr_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools

source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

echo "============================================================"
echo "DPO STEERING VECTOR - LAYER 20 - LOW LR (5e-4) - 3 EPOCHS"
echo "============================================================"

python -u elicitation/finetune_dpo_steering_vector.py \
    google/gemma-3-27b-it \
    elicitation/outputs/dpo_pairs_full.jsonl \
    /workspace-vast/annas/models/gemma3-27b-dpo-steering-vector-L20-lowlr-3ep \
    --layer 20 \
    --lr 5e-4 \
    --num_epochs 3 \
    --beta 0.1 \
    --alpha 256 \
    --init_scale 0.01 \
    --max_length 2048 \
    --log_every 10

echo "Done!"
