#!/bin/bash
#SBATCH --job-name=ortho_reg_probe
#SBATCH --output=probes/logs/ortho_reg_probe_%A_%a.log
#SBATCH --error=probes/logs/ortho_reg_probe_%A_%a.err
#SBATCH --partition=general
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-29%10  # 6 lambdas × 5 layers = 30 jobs, max 10 concurrent

# Lambda sweep for orthogonality-regularized probes
# Usage: sbatch probes/scripts/slurm_jobs/training/train_orthogonal_regularized_sweep.sh

set -e

echo "========================================"
echo "ORTHOGONAL REGULARIZED PROBE TRAINING"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo ""

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Create output directories
mkdir -p probes/results/orthogonal_regularized_probes
mkdir -p probes/logs

# Configuration
DATA_PATH=${DATA_PATH:-"outputs/data/activations/texts_combined.h5"}
K=${K:-20}  # Number of neutral PCs used
NEUTRAL_PCS_DIR=${NEUTRAL_PCS_DIR:-"probes/results/neutral_pcs"}
OUTPUT_DIR=${OUTPUT_DIR:-"probes/results/orthogonal_regularized_probes"}

# Arrays of hyperparameters
LAYERS=(10 15 20 25 30)
LAMBDAS=(0.0 0.01 0.1 1.0 10.0 100.0)

# Compute which layer and lambda to use
N_LAYERS=${#LAYERS[@]}
N_LAMBDAS=${#LAMBDAS[@]}

LAYER_IDX=$((SLURM_ARRAY_TASK_ID / N_LAMBDAS))
LAMBDA_IDX=$((SLURM_ARRAY_TASK_ID % N_LAMBDAS))

LAYER=${LAYERS[$LAYER_IDX]}
LAMBDA=${LAMBDAS[$LAMBDA_IDX]}

echo "Configuration:"
echo "  Layer: $LAYER"
echo "  Lambda: $LAMBDA"
echo "  k: $K"
echo "  Data: $DATA_PATH"
echo ""

# Determine neutral PCs path
if [ "$LAMBDA" = "0.0" ]; then
    # Baseline - no orthogonality regularization
    echo "Training baseline probe (no orthogonality regularization)"
    NEUTRAL_PCS_ARG=""
else
    NEUTRAL_PCS_PATH="${NEUTRAL_PCS_DIR}/layer_${LAYER}_neutral_pcs_k${K}.npy"

    if [ ! -f "$NEUTRAL_PCS_PATH" ]; then
        echo "ERROR: Neutral PCs not found at $NEUTRAL_PCS_PATH"
        echo "Please run compute_neutral_pcs.sh first:"
        echo "  sbatch probes/scripts/slurm_jobs/dimensionality_reduction/compute_neutral_pcs.sh"
        exit 1
    fi

    echo "Using neutral PCs: $NEUTRAL_PCS_PATH"
    NEUTRAL_PCS_ARG="--neutral-pcs $NEUTRAL_PCS_PATH"
fi

# Run training
python probes/scripts/training/train_orthogonal_regularized_probe.py \
    --data $DATA_PATH \
    --layer $LAYER \
    $NEUTRAL_PCS_ARG \
    --lambda-ortho $LAMBDA \
    --max-epochs 500 \
    --patience 10 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --weight-decay 1.0 \
    --output-dir $OUTPUT_DIR \
    --device cuda \
    --seed 42

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Training completed successfully!"
    echo "  Layer: $LAYER, Lambda: $LAMBDA"
fi

exit $EXIT_CODE
