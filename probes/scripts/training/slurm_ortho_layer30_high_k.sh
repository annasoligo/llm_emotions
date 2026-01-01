#!/bin/bash
#SBATCH --job-name=ortho_l30_high_k
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/ortho_layer30_high_k_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/ortho_layer30_high_k_%A.err
#SBATCH --time=48:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Train orthogonal emotion probes at layer 30 with increasing K values
# K values: 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
# Using very high ortho weight (100k) to strongly enforce orthogonality
# Stops when convergence fails

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
MAX_K=50  # Will test up to this if all converge

echo "========================================"
echo "Layer 30 High-K Orthogonal Probe Search"
echo "Ortho weight: $ORTHO_WEIGHT"
echo "Testing K: 5, 10, 15, 20, 25, 30, 35, 40, 45, 50"
echo "Will stop when convergence fails"
echo "========================================"
echo ""

# Test individual K values to find the breaking point
for K in 5 10 15 20 25 30 35 40 45 50; do
    echo "========================================"
    echo "Training Layer $LAYER with K=$K"
    echo "========================================"

    python probes/scripts/training/train_multi_orthogonal_text_probes.py \
        --layer $LAYER \
        --n-sets $K \
        --ortho-weight $ORTHO_WEIGHT \
        --max-epochs 300 \
        --patience 30 \
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
        echo "Maximum converged K is likely $(($K - 5))"
        exit 1
    fi

    # Check if convergence was achieved by looking at the output
    echo ""
    echo "✓ Completed K=$K"
    echo ""

    # Brief pause between runs
    sleep 5
done

echo "========================================"
echo "All K values up to $MAX_K completed successfully!"
echo "Consider testing higher K values."
echo "========================================"
