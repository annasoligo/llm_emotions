#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=oracle_filter
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/%j.out
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/%j.err

# GPU job for oracle filtering
# Requires Gemma-3-27B model (~55GB in bfloat16)

# Load authentication
source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME=/workspace-vast/pretrained_ckpts

# Activate environment
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Run oracle filtering
python probes/scripts/filter_conversations_with_oracle.py \
    --input outputs/data/conversations2.jsonl \
    --output outputs/data/conversations2_filtered.jsonl \
    --base_model google/gemma-3-27b-it \
    --oracle_adapter annasoli/gemma-3-27b-activation-oracle-BS32 \
    --device cuda
