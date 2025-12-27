#!/bin/bash
#SBATCH --job-name=emotion_probe
#SBATCH --output=slurm_logs/emotion_probe_layer%a.log
#SBATCH --error=slurm_logs/emotion_probe_layer%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=general

# Array job to train probes on multiple layers
# Usage: sbatch --array=20,30,40,50 scripts/slurm_jobs/train_emotion_probe.sh

# Default to layer from array task ID if not specified
LAYER=${LAYER:-$SLURM_ARRAY_TASK_ID}

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Layer: $LAYER"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "=========================================="

# Navigate to repo root
cd /workspace-vast/annas/git/research-tools || exit 1

# Activate virtual environment
source .venv/bin/activate

# Create output directories
mkdir -p results/emotion_probes
mkdir -p slurm_logs

# Default arguments (can be overridden by environment variables)
DATA_PATH=${DATA_PATH:-"data/activations/texts.h5"}
TIERS=${TIERS:-""}  # Empty means all tiers
MAX_SAMPLES=${MAX_SAMPLES:-""}  # Empty means no balancing
TEST_SIZE=${TEST_SIZE:-"0.2"}
LEARNING_RATE=${LEARNING_RATE:-"0.001"}
BATCH_SIZE=${BATCH_SIZE:-"64"}
MAX_EPOCHS=${MAX_EPOCHS:-"500"}
PATIENCE=${PATIENCE:-"10"}
WEIGHT_DECAY=${WEIGHT_DECAY:-"1.0"}  # L2 regularization
L1_LAMBDA=${L1_LAMBDA:-"0.0"}  # L1 regularization (disabled by default)
N_COMPONENTS=${N_COMPONENTS:-""}  # Number of PCs to use (empty = all)
OUTPUT_DIR=${OUTPUT_DIR:-"results/emotion_probes"}
SEED=${SEED:-"42"}
USE_CPCA=${USE_CPCA:-""}  # Set to "1" to enable cPCA
CPCA_RESULTS=${CPCA_RESULTS:-"probes/results/cpca_tier_data/google/gemma-3-27b-it_cpca.npz"}

# Build command
CMD="python probes/scripts/train_emotion_probe.py \
    --data $DATA_PATH \
    --layer $LAYER \
    --test-size $TEST_SIZE \
    --learning-rate $LEARNING_RATE \
    --batch-size $BATCH_SIZE \
    --max-epochs $MAX_EPOCHS \
    --patience $PATIENCE \
    --weight-decay $WEIGHT_DECAY \
    --l1-lambda $L1_LAMBDA \
    --output-dir $OUTPUT_DIR \
    --seed $SEED \
    --device cuda"

# Add optional arguments
if [ -n "$TIERS" ]; then
    CMD="$CMD --tiers $TIERS"
fi

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max-samples $MAX_SAMPLES"
fi

if [ "$USE_CPCA" = "1" ]; then
    CMD="$CMD --use-cpca --cpca-results $CPCA_RESULTS"
fi

if [ -n "$N_COMPONENTS" ]; then
    CMD="$CMD --n-components $N_COMPONENTS"
fi

echo ""
echo "Command:"
echo "$CMD"
echo ""

# Run training
eval $CMD

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
