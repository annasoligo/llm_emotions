#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --job-name=merge_L2025
#SBATCH --output=/workspace-vast/annas/logs/merge_layers20-25_%j.out
#SBATCH --error=/workspace-vast/annas/logs/merge_layers20-25_%j.err
#SBATCH --time=01:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME="/workspace-vast/annas/.cache/huggingface"

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python elicitation/merge_and_upload_lora.py \
    --base-model "google/gemma-3-27b-it" \
    --adapter "/workspace-vast/annas/models/gemma3-27b-dpo-r64-layers20-25-2ep/2026-01-18_12-55-37" \
    --output-repo "annasoli/gemma3-27b-dpo-r64-layers20-25-2ep-merged" \
    --local-dir "/workspace-vast/annas/models/gemma3-27b-dpo-r64-layers20-25-2ep-merged"

echo "Merge and upload complete!"
