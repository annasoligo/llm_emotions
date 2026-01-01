#!/bin/bash
#SBATCH --job-name=k50_l28
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/train_k50_layer28_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/train_k50_layer28_%A.err
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

cd /workspace-vast/annas/git/research-tools

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Activated venv"
fi

echo "========================================"
echo "Training K=50 Layer 28"
echo "========================================"

python probes/scripts/training/train_multi_orthogonal_conversation_probes.py \
    --layer 28 \
    --n-sets 50 \
    --ortho-weight 100000.0 \
    --max-epochs 300 \
    --patience 3 \
    --learning-rate 0.001 \
    --batch-size 32 \
    --seed 42 \
    --device cuda

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training failed for layer 28"
    exit 1
fi

echo "✓ Layer 28 completed!"
