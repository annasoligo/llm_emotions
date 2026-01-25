#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --job-name=dpo_steer_vec
#SBATCH --output=/workspace-vast/annas/logs/dpo_steering_vector_%j.out
#SBATCH --error=/workspace-vast/annas/logs/dpo_steering_vector_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools

# Use believe-it-or-not venv for transformers/torch
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

echo "============================================================"
echo "DPO STEERING VECTOR TRAINING"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================================"

# Only ~5377 trainable parameters!
# Use max_length=512 to include response tokens
python -u elicitation/finetune_dpo_steering_vector.py \
    google/gemma-3-27b-it \
    elicitation/outputs/dpo_pairs_full.jsonl \
    /workspace-vast/annas/models/gemma3-27b-dpo-steering-vector-L20-alpha256 \
    --layer 20 \
    --lr 1e-3 \
    --num_epochs 2 \
    --beta 0.1 \
    --alpha 256 \
    --init_scale 0.01 \
    --max_length 2048 \
    --log_every 10

echo "Done!"
