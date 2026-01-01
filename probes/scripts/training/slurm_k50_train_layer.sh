#!/bin/bash
# Parameterized SLURM script for training K=50 orthogonal conversation probes
# Usage: sbatch --export=LAYER=<layer_number> slurm_k50_train_layer.sh
# Example: sbatch --export=LAYER=30 slurm_k50_train_layer.sh
#
# Or use the convenience wrapper script:
#   ./submit_k50_layer.sh 30

#SBATCH --job-name=k50_l%LAYER%
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/train_k50_layer%LAYER%_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/train_k50_layer%LAYER%_%A.err
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Validate LAYER environment variable
if [ -z "$LAYER" ]; then
    echo "ERROR: LAYER environment variable not set"
    echo "Usage: sbatch --export=LAYER=<layer_number> slurm_k50_train_layer.sh"
    echo "Example: sbatch --export=LAYER=30 slurm_k50_train_layer.sh"
    exit 1
fi

cd /workspace-vast/annas/git/research-tools

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Activated venv"
fi

echo "========================================"
echo "Training K=50 Layer ${LAYER}"
echo "========================================"

python probes/scripts/training/train_multi_orthogonal_conversation_probes.py \
    --layer ${LAYER} \
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
    echo "ERROR: Training failed for layer ${LAYER}"
    exit 1
fi

echo "✓ Layer ${LAYER} completed!"
