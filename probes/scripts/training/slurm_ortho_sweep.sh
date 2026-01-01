#!/bin/bash
#SBATCH --job-name=ortho_sweep_k5
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/ortho_sweep_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/ortho_sweep_%A.err
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Train orthogonal emotion probes across layers 0-50 with K=1-5
# Using very high ortho weight (100k) to strongly enforce orthogonality

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

# Training parameters
ORTHO_WEIGHT=100000.0
MAX_K=5
LAYERS="0 5 10 15 20 25 30 35 40 45 50"

echo "========================================"
echo "Starting Orthogonal Probe Training Sweep"
echo "Ortho weight: $ORTHO_WEIGHT"
echo "Max K: $MAX_K"
echo "Layers: $LAYERS"
echo "========================================"
echo ""

for LAYER in $LAYERS; do
    echo "========================================"
    echo "Training Layer $LAYER with K=1-${MAX_K}"
    echo "========================================"

    python probes/scripts/training/train_multi_orthogonal_text_probes.py \
        --layer $LAYER \
        --search-k \
        --max-sets $MAX_K \
        --ortho-weight $ORTHO_WEIGHT \
        --max-epochs 200 \
        --patience 20 \
        --learning-rate 0.001 \
        --batch-size 32 \
        --seed 42 \
        --device cuda \
        --convergence-threshold 0.001 \
        --min-accuracy-threshold 0.25 \
        --accuracy-degradation-threshold 0.05

    if [ $? -ne 0 ]; then
        echo "ERROR: Training failed for layer $LAYER"
        exit 1
    fi

    echo ""
    echo "✓ Completed layer $LAYER"
    echo ""
done

echo "========================================"
echo "All layers completed successfully!"
echo "========================================"
