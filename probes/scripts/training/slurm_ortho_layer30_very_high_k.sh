#!/bin/bash
#SBATCH --job-name=ortho_l30_vhigh
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/ortho_layer30_very_high_k_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/ortho_layer30_very_high_k_%A.err
#SBATCH --time=48:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Train orthogonal emotion probes at layer 30 with very high K values
# K values: 60, 70, 80, 90, 100
# Using very high ortho weight (100k) to strongly enforce orthogonality
# Parallel to job testing K=35-50

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
LAYER=30

echo "========================================"
echo "Layer 30 Very High-K Orthogonal Probe Test"
echo "Ortho weight: $ORTHO_WEIGHT"
echo "Testing K: 120, 140, 160 with patience 3"
echo "Parallel to job testing K=35-50"
echo "========================================"
echo ""

# Test very high K values
for K in 120 140 160; do
    echo "========================================"
    echo "Training Layer $LAYER with K=$K"
    echo "========================================"

    python probes/scripts/training/train_multi_orthogonal_text_probes.py \
        --layer $LAYER \
        --n-sets $K \
        --ortho-weight $ORTHO_WEIGHT \
        --max-epochs 300 \
        --patience 3 \
        --learning-rate 0.001 \
        --batch-size 32 \
        --seed 42 \
        --device cuda \
        --convergence-threshold 0.01 \
        --min-accuracy-threshold 0.25 \
        --accuracy-degradation-threshold 0.05

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Training failed for K=$K"
        echo "Maximum converged K is likely less than $K"
        exit 1
    fi

    echo ""
    echo "✓ Completed K=$K"
    echo ""

    # Brief pause between runs
    sleep 5
done

echo "========================================"
echo "All very high K values completed successfully!"
echo "Layer 30 can support at least K=160 orthogonal probe sets!"
echo "========================================"
