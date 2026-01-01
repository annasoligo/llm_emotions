#!/bin/bash
#SBATCH --job-name=alpaca_activations
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/alpaca_activations_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/alpaca_activations_%A.err
#SBATCH --time=04:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Load environment
cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate venv
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Activated venv"
else
    echo "⚠ No venv found at .venv/bin/activate"
fi

# Set HuggingFace cache
export HF_HOME=/workspace-vast/pretrained_ckpts

echo "=============================================================================="
echo "COLLECTING ALPACA BASELINE ACTIVATIONS"
echo "=============================================================================="

python probes/scripts/compute_baseline_activations_from_jsonl.py \
    --input data/alpaca_responses.jsonl \
    --output data/baselines/alpaca_gemma27b \
    --model-name unsloth/gemma-3-27b-it \
    --layers 0-62 \
    --device cuda

echo "✓ Complete"
