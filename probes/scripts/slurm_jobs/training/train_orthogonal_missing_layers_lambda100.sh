#!/bin/bash
#SBATCH --job-name=ortho_missing_l100
#SBATCH --output=probes/logs/ortho_missing_l100_%A_%a.log
#SBATCH --error=probes/logs/ortho_missing_l100_%A_%a.err
#SBATCH --array=0-8%3
#SBATCH --partition=general
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -e

echo "========================================"
echo "TRAIN ORTHOGONAL PROBES: MISSING LAYERS, λ=100.0"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo ""

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Create output directories
mkdir -p outputs/probes/emotion_probes/text_based/orthogonal_regularized/lambda_100.0
mkdir -p probes/logs

# Configuration
DATA_PATH="outputs/data/activations/texts_combined.h5"
K=20
LAMBDA=100.0
NEUTRAL_PCS_DIR="probes/results/neutral_pcs"
OUTPUT_DIR="outputs/probes/emotion_probes/text_based/orthogonal_regularized/lambda_100.0"

# Map array task ID to missing layers
MISSING_LAYERS=(21 22 25 29 30 32 34 39 40)
LAYER=${MISSING_LAYERS[$SLURM_ARRAY_TASK_ID]}

echo "Configuration:"
echo "  Layer: $LAYER"
echo "  Lambda: $LAMBDA"
echo "  k: $K"
echo "  Data: $DATA_PATH"
echo ""

# Check if neutral PCs exist for this layer
NEUTRAL_PCS_PATH="${NEUTRAL_PCS_DIR}/layer_${LAYER}_neutral_pcs_k${K}.npy"
if [ ! -f "$NEUTRAL_PCS_PATH" ]; then
    echo "ERROR: Neutral PCs not found at $NEUTRAL_PCS_PATH"
    exit 1
fi

echo "Using neutral PCs: $NEUTRAL_PCS_PATH"
echo ""

# Train probe
python probes/scripts/training/train_orthogonal_regularized_probe.py \
    --data $DATA_PATH \
    --layer $LAYER \
    --neutral-pcs $NEUTRAL_PCS_PATH \
    --lambda-ortho $LAMBDA \
    --output-dir $OUTPUT_DIR \
    --seed 42

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Probe trained successfully!"
    echo "  Layer: $LAYER"
    echo "  Lambda: $LAMBDA"
    echo "  Output: $OUTPUT_DIR/"
fi

exit $EXIT_CODE
