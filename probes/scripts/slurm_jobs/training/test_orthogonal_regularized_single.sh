#!/bin/bash
#SBATCH --job-name=ortho_test
#SBATCH --output=probes/logs/ortho_test_%j.log
#SBATCH --error=probes/logs/ortho_test_%j.err
#SBATCH --partition=general
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Quick test of orthogonal regularized probe on single layer
# Usage: sbatch probes/scripts/slurm_jobs/training/test_orthogonal_regularized_single.sh

set -e

echo "========================================"
echo "TESTING ORTHOGONAL REGULARIZED PROBES"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo ""

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Create output directories
mkdir -p probes/results/neutral_pcs_test
mkdir -p probes/results/orthogonal_regularized_probes_test
mkdir -p probes/logs

# Configuration
DATA_PATH="outputs/data/activations/texts_combined.h5"
LAYER=20
K=20
LAMBDA=1.0

echo "Configuration:"
echo "  Data: $DATA_PATH"
echo "  Layer: $LAYER"
echo "  k: $K"
echo "  Lambda: $LAMBDA"
echo ""

# Step 1: Compute neutral PCs for this layer
echo "========================================"
echo "STEP 1: Computing neutral PCs"
echo "========================================"

python probes/scripts/dimensionality_reduction/compute_neutral_pcs_for_regularization.py \
    --data $DATA_PATH \
    --layer $LAYER \
    --k $K \
    --output probes/results/neutral_pcs_test \
    --seed 42

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to compute neutral PCs"
    exit 1
fi

echo ""
echo "✓ Neutral PCs computed"
echo ""

# Step 2: Train baseline probe (no regularization)
echo "========================================"
echo "STEP 2: Training baseline probe (λ=0)"
echo "========================================"

python probes/scripts/training/train_orthogonal_regularized_probe.py \
    --data $DATA_PATH \
    --layer $LAYER \
    --lambda-ortho 0.0 \
    --max-epochs 100 \
    --patience 10 \
    --batch-size 64 \
    --output-dir probes/results/orthogonal_regularized_probes_test/baseline \
    --device cuda \
    --seed 42

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to train baseline probe"
    exit 1
fi

echo ""
echo "✓ Baseline probe trained"
echo ""

# Step 3: Train orthogonality-regularized probe
echo "========================================"
echo "STEP 3: Training orthogonal probe (λ=$LAMBDA)"
echo "========================================"

python probes/scripts/training/train_orthogonal_regularized_probe.py \
    --data $DATA_PATH \
    --layer $LAYER \
    --neutral-pcs probes/results/neutral_pcs_test/layer_${LAYER}_neutral_pcs_k${K}.npy \
    --lambda-ortho $LAMBDA \
    --max-epochs 100 \
    --patience 10 \
    --batch-size 64 \
    --output-dir probes/results/orthogonal_regularized_probes_test/lambda_${LAMBDA} \
    --device cuda \
    --seed 42

EXIT_CODE=$?

echo ""
echo "========================================"
echo "TEST COMPLETE"
echo "========================================"
echo "Completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ All steps completed successfully!"
    echo ""
    echo "Results:"
    echo "  Neutral PCs: probes/results/neutral_pcs_test/"
    echo "  Baseline probe: probes/results/orthogonal_regularized_probes_test/baseline/"
    echo "  Orthogonal probe: probes/results/orthogonal_regularized_probes_test/lambda_${LAMBDA}/"
    echo ""
    echo "Check the summary files:"
    echo "  cat probes/results/orthogonal_regularized_probes_test/baseline/*_summary.txt"
    echo "  cat probes/results/orthogonal_regularized_probes_test/lambda_${LAMBDA}/*_summary.txt"
    echo ""
    echo "Next step: Run full sweep"
    echo "  sbatch probes/scripts/slurm_jobs/training/train_orthogonal_regularized_sweep.sh"
fi

exit $EXIT_CODE
