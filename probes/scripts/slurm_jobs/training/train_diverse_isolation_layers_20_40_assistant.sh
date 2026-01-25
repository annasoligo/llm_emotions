#!/bin/bash
#SBATCH --job-name=diverse_20_40_asst
#SBATCH --output=/workspace-vast/annas/logs/%j.out
#SBATCH --error=/workspace-vast/annas/logs/%j.err
#SBATCH --partition=general
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Train ASSISTANT probes for layers 20-40 with lambda=10

set -e

echo "========================================"
echo "DIVERSE ISOLATION ASSISTANT TRAINING (LAYERS 20-40, LAMBDA=10)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo ""

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Configuration
ISOLATION_TYPE="assistant"
LAMBDA=10
NUM_EPOCHS=20
BATCH_SIZE=32
LR=1e-3
WEIGHT_DECAY=10.0
OUTPUT_DIR="outputs/probes/diverse_isolation"

echo "Configuration:"
echo "  Isolation type: $ISOLATION_TYPE"
echo "  Layers: 20-40 (21 layers)"
echo "  Lambda: $LAMBDA"
echo "  Epochs: $NUM_EPOCHS"
echo "  Weight decay: $WEIGHT_DECAY"
echo ""

# Train for each layer
for LAYER in {20..40}; do
    echo ""
    echo "========================================"
    echo "LAYER $LAYER"
    echo "========================================"

    python3 probes/scripts/training/train_diverse_isolation_probe.py \
        --isolation-type $ISOLATION_TYPE \
        --layer $LAYER \
        --lambda-reg $LAMBDA \
        --num-epochs $NUM_EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --weight-decay $WEIGHT_DECAY \
        --output $OUTPUT_DIR \
        --seed 42

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Training failed for layer=$LAYER"
        exit 1
    fi

    echo "✓ Layer=$LAYER completed"
done

echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"
echo ""
printf "%-6s | %-8s | %-8s\\n" "Layer" "Val Acc" "Test Acc"
printf "%-6s-+-%-8s-+-%-8s\\n" "------" "--------" "--------"

for LAYER in {20..40}; do
    RESULT_DIR="${OUTPUT_DIR}/${ISOLATION_TYPE}_layer${LAYER}_lambda${LAMBDA}"
    if [ -f "${RESULT_DIR}/results.json" ]; then
        python3 -c "
import json
with open('${RESULT_DIR}/results.json', 'r') as f:
    results = json.load(f)
val_acc = results['best_val_acc']
test_acc = results['test_results']['accuracy']
print(f'{$LAYER:6d} | {val_acc:8.4f} | {test_acc:8.4f}')
"
    fi
done

echo ""
echo "========================================"
echo "✓ ASSISTANT TRAINING COMPLETE"
echo "========================================"
echo "Completed at: $(date)"
echo "Results saved to: $OUTPUT_DIR"
