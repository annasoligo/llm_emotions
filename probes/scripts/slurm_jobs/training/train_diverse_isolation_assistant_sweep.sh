#!/bin/bash
#SBATCH --job-name=diverse_asst
#SBATCH --output=/workspace-vast/annas/logs/%j.out
#SBATCH --error=/workspace-vast/annas/logs/%j.err
#SBATCH --partition=general
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Train diverse isolation ASSISTANT probes across multiple layers and lambda values

set -e

echo "========================================"
echo "DIVERSE ISOLATION ASSISTANT TRAINING SWEEP"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo ""

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Configuration
ISOLATION_TYPE="assistant"
LAYERS=(0 10 20 30 40 50 60)
LAMBDAS=(0 10 100)
NUM_EPOCHS=20
BATCH_SIZE=32
LR=1e-3
WEIGHT_DECAY=10.0
OUTPUT_DIR="outputs/probes/diverse_isolation"

echo "Configuration:"
echo "  Isolation type: $ISOLATION_TYPE"
echo "  Layers: ${LAYERS[@]}"
echo "  Lambdas: ${LAMBDAS[@]}"
echo "  Epochs: $NUM_EPOCHS"
echo "  Weight decay: $WEIGHT_DECAY"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Train for each layer and lambda
for LAYER in "${LAYERS[@]}"; do
    echo ""
    echo "========================================"
    echo "LAYER $LAYER"
    echo "========================================"

    for LAMBDA in "${LAMBDAS[@]}"; do
        echo ""
        echo "--- Lambda=$LAMBDA ---"

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
            echo "ERROR: Training failed for layer=$LAYER, lambda=$LAMBDA"
            exit 1
        fi

        echo "✓ Layer=$LAYER, Lambda=$LAMBDA completed"
    done
done

echo ""
echo "========================================"
echo "SUMMARY ACROSS ALL CONFIGURATIONS"
echo "========================================"
echo ""
printf "%-6s | %-8s | %-8s | %-8s\n" "Layer" "Lambda" "Val Acc" "Test Acc"
printf "%-6s-+-%-8s-+-%-8s-+-%-8s\n" "------" "--------" "--------" "--------"

for LAYER in "${LAYERS[@]}"; do
    for LAMBDA in "${LAMBDAS[@]}"; do
        RESULT_DIR="${OUTPUT_DIR}/${ISOLATION_TYPE}_layer${LAYER}_lambda${LAMBDA}"
        if [ -f "${RESULT_DIR}/results.json" ]; then
            python3 -c "
import json
with open('${RESULT_DIR}/results.json', 'r') as f:
    results = json.load(f)
val_acc = results['best_val_acc']
test_acc = results['test_results']['accuracy']
print(f'{$LAYER:6d} | {$LAMBDA:8.1f} | {val_acc:8.4f} | {test_acc:8.4f}')
"
        fi
    done
done

echo ""
echo "========================================"
echo "✓ ASSISTANT TRAINING SWEEP COMPLETE"
echo "========================================"
echo "Completed at: $(date)"
echo "Results saved to: $OUTPUT_DIR"
