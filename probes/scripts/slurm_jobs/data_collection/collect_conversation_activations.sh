#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=conv_acts
#SBATCH --output=/workspace-vast/annas/logs/%j.out
#SBATCH --error=/workspace-vast/annas/logs/%j.err

# Collect activations for multi-turn conversations
# Uses chat tokenization and extracts regional + global activations

# Load authentication
source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME=/workspace-vast/pretrained_ckpts

# Activate environment
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

python -m probes.scripts.collect_activations \
    --input outputs/data/conversations2.jsonl \
    --output outputs/data/activations/conversations2.h5 \
    --model google/gemma-3-27b-it \
    --data_type conversations
