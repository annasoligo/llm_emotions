#!/bin/bash
#SBATCH --job-name=ortho_l30_k50_gs
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/ortho_layer30_k50_gramschmidt_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/ortho_layer30_k50_gramschmidt_%A.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Train K=50 orthogonal emotion probes at layer 30 with Gram-Schmidt re-orthogonalization

cd /workspace-vast/annas/git/research-tools

# Activate virtual environment
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Activated venv"
else
    echo "⚠ No venv found"
fi

echo "========================================"
echo "Training Layer 30, K=50 with Gram-Schmidt"
echo "Ortho weight: 100000.0"
echo "Patience: 3"
echo "========================================"
echo ""

LAYER=30
K=50
ORTHO_WEIGHT=100000.0

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
    echo "ERROR: Training failed for layer $LAYER, K=$K"
    exit 1
fi

echo ""
echo "========================================"
echo "✓ Layer $LAYER, K=$K with Gram-Schmidt completed!"
echo "========================================"
