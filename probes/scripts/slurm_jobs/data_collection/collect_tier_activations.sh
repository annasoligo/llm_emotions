#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=comb_probes
#SBATCH --output=/workspace-vast/annas/logs/%j.out
#SBATCH --error=/workspace-vast/annas/logs/%j.err

# GPU finetuning job - requires 1 GPU
# Note: --gres=gpu:1 requests 1 GPU

# Load authentication
source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME=/workspace-vast/pretrained_ckpts

# Activate environment if needed
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

python -m probes.scripts.collect_activations \
    --input data/texts.jsonl \
    --output data/activations/texts.h5 \
    --model google/gemma-3-27b-it \
    --data_type texts \
    --start_token 20