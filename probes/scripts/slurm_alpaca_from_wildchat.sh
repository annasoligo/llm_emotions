#!/bin/bash
#SBATCH --job-name=alpaca_via_wildchat
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/alpaca_via_wildchat_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/alpaca_via_wildchat_%A.err
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

# Install h5py if needed
pip show h5py > /dev/null 2>&1 || pip install h5py

# Run activation extraction using the wildchat script but with alpaca data
echo "=============================================================================="
echo "COLLECTING ALPACA BASELINE ACTIVATIONS VIA WILDCHAT SCRIPT"
echo "=============================================================================="

python probes/scripts/compute_wildchat_baseline_activations.py \
    --wildchat-cache data/alpaca_responses.jsonl \
    --output-dir data/baselines/alpaca_gemma27b_v2 \
    --layers all \
    --num-samples 512

echo "✓ Complete"
