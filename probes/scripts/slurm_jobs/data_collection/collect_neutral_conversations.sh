#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=collect_neutral_convs
#SBATCH --output=/workspace-vast/annas/logs/%j.out
#SBATCH --error=/workspace-vast/annas/logs/%j.err

# Collect activations for neutral conversations

source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME=/workspace-vast/pretrained_ckpts

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

echo "Collecting activations for neutral conversations..."
python -m probes.scripts.collect_activations \
    --input data/conversations2_neutral.jsonl \
    --output data/activations/conversations2_neutral.h5 \
    --model google/gemma-3-27b-it \
    --data_type conversations

echo ""
echo "Done! Saved to data/activations/conversations2_neutral.h5"
