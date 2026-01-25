#!/bin/bash
#SBATCH --job-name=neutral_pcs_20_40
#SBATCH --output=probes/logs/neutral_pcs_20_40_%j.log
#SBATCH --error=probes/logs/neutral_pcs_20_40_%j.err
#SBATCH --partition=general
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -e

echo "========================================"
echo "COMPUTE NEUTRAL PCs FOR LAYERS 20-40"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo ""

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Create output directories
mkdir -p probes/results/neutral_pcs
mkdir -p probes/logs

# Configuration
DATA_PATH="outputs/data/activations/texts_combined.h5"
K=20  # Number of neutral PCs to extract
N_PCS_ALL=100  # Number of PCs for ratio analysis
LAYERS="20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40"
OUTPUT_DIR="probes/results/neutral_pcs"

echo "Configuration:"
echo "  Data: $DATA_PATH"
echo "  k (neutral PCs): $K"
echo "  n_pcs_all: $N_PCS_ALL"
echo "  Layers: $LAYERS"
echo "  Output: $OUTPUT_DIR"
echo ""

# Run computation
python probes/scripts/dimensionality_reduction/compute_neutral_pcs_for_regularization.py \
    --data $DATA_PATH \
    --layers $LAYERS \
    --k $K \
    --n-pcs-all $N_PCS_ALL \
    --output $OUTPUT_DIR \
    --seed 42

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Neutral PCs computed successfully for layers 20-40!"
    echo "  Output: $OUTPUT_DIR/"
fi

exit $EXIT_CODE
