#!/bin/bash
#SBATCH --job-name=ortho_gs_sweep
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/ortho_layer30_gramschmidt_sweep_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/ortho_layer30_gramschmidt_sweep_%A.err
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Train orthogonal emotion probes at layer 30 with Gram-Schmidt
# Test K = 10, 20, 30, 40

cd /workspace-vast/annas/git/research-tools

# Activate virtual environment
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Activated venv"
else
    echo "⚠ No venv found"
fi

echo "========================================"
echo "Training Layer 30 with Gram-Schmidt"
echo "K values: 10, 20, 30, 40"
echo "Ortho weight: 100000.0"
echo "Patience: 3"
echo "========================================"
echo ""

LAYER=30
ORTHO_WEIGHT=100000.0

for K in 10 20 30 40; do
    echo ""
    echo "========================================"
    echo "Training K=$K with Gram-Schmidt"
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
        --gram-schmidt

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Training failed for K=$K"
        exit 1
    fi

    # Rename output to include gram_schmidt suffix
    ORIGINAL_NAME="probes/emotion_probes/text_based/multi_orthogonal/probe_k${K}_layer${LAYER}_ortho${ORTHO_WEIGHT}.pkl"
    NEW_NAME="probes/emotion_probes/text_based/multi_orthogonal/probe_k${K}_layer${LAYER}_ortho${ORTHO_WEIGHT}_gramschmidt.pkl"

    if [ -f "$ORIGINAL_NAME" ]; then
        mv "$ORIGINAL_NAME" "$NEW_NAME"
        echo "✓ Renamed to: $NEW_NAME"
    fi

    echo "✓ Completed K=$K"
done

echo ""
echo "========================================"
echo "✓ All Gram-Schmidt training completed!"
echo "========================================"
