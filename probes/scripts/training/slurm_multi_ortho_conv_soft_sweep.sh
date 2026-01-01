#!/bin/bash
#SBATCH --job-name=conv_soft_sweep
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/multi_ortho_conv_soft_sweep_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/multi_ortho_conv_soft_sweep_%A.err
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Train K orthogonal user/assistant pairs with SOFT constraint
# Test K = 10, 20, 30, 40, 50

cd /workspace-vast/annas/git/research-tools

# Activate virtual environment
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Activated venv"
else
    echo "⚠ No venv found"
fi

echo "========================================"
echo "Multi-Orthogonal Conversation Probes"
echo "SOFT CONSTRAINT (loss-based)"
echo "K values: 10, 20, 30, 40, 50"
echo "Ortho weight: 100000.0"
echo "========================================"
echo ""

LAYER=30
ORTHO_WEIGHT=100000.0

for K in 10 20 30 40 50; do
    echo ""
    echo "========================================"
    echo "Training K=$K (SOFT)"
    echo "========================================"

    python probes/scripts/training/train_multi_orthogonal_conversation_probes.py \
        --layer $LAYER \
        --n-sets $K \
        --ortho-weight $ORTHO_WEIGHT \
        --max-epochs 300 \
        --patience 3 \
        --learning-rate 0.001 \
        --batch-size 32 \
        --seed 42 \
        --device cuda

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Training failed for K=$K"
        exit 1
    fi

    echo "✓ Completed K=$K"
done

echo ""
echo "========================================"
echo "✓ All soft constraint training completed!"
echo "========================================"
