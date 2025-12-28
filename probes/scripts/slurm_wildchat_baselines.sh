#!/bin/bash
#SBATCH --job-name=wildchat_baseline_all
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/wildchat_baseline_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/wildchat_baseline_%A.err
#SBATCH --time=03:00:00
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

# Run activation extraction for ALL layers in single pass
echo "Starting WildChat baseline activation extraction..."
echo "Layers: ${1:-all}"
echo "Num samples: ${2:-512}"

python probes/scripts/compute_wildchat_baseline_activations.py \
    --layers "${1:-all}" \
    --num-samples ${2:-512}

echo "✓ Activation extraction complete for all layers"
