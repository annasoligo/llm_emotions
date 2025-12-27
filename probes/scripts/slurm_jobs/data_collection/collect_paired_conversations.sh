#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=collect_paired_convs
#SBATCH --output=/workspace-vast/annas/logs/%j.out
#SBATCH --error=/workspace-vast/annas/logs/%j.err

# Collect paired conversation activations (emotional + neutral)

set -e

# Load authentication
source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME=/workspace-vast/pretrained_ckpts

# Activate environment
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "========================================"
echo "Collecting Paired Conversation Activations"
echo "========================================"
echo

# Run collection
python -m probes.scripts.collect_paired_conversation_activations \
    --emotional data/conversations2.jsonl \
    --neutral data/conversations2_neutral.jsonl \
    --output data/activations/conversations2_paired.h5 \
    --model google/gemma-3-27b-it

echo
echo "========================================"
echo "Collection complete!"
echo "========================================"
echo "Output: data/activations/conversations2_paired.h5"
