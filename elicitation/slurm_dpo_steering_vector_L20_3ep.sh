#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --job-name=dpo_sv_L20_3ep
#SBATCH --output=/workspace-vast/annas/logs/dpo_sv_L20_3ep_%j.out
#SBATCH --error=/workspace-vast/annas/logs/dpo_sv_L20_3ep_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools

source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

echo "============================================================"
echo "DPO STEERING VECTOR - LAYER 20 - 3 EPOCHS (lr=1e-3)"
echo "============================================================"

python -u elicitation/finetune_dpo_steering_vector.py \
    google/gemma-3-27b-it \
    elicitation/outputs/dpo_pairs_full.jsonl \
    /workspace-vast/annas/models/gemma3-27b-dpo-steering-vector-L20-3ep \
    --layer 20 \
    --lr 1e-3 \
    --num_epochs 3 \
    --beta 0.1 \
    --alpha 256 \
    --init_scale 0.01 \
    --max_length 2048 \
    --log_every 10

echo "Done!"
