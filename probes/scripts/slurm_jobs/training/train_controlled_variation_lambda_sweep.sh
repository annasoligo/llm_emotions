#!/bin/bash
#SBATCH --job-name=cv_lambda_sweep
#SBATCH --output=/workspace-vast/annas/logs/%j.out
#SBATCH --error=/workspace-vast/annas/logs/%j.err
#SBATCH --partition=general
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Train orthogonal regularized probes on controlled variation data
# Tests lambda values: 0 (baseline), 10, 100, 1000
# Layer: 30 (high entanglement in original cPCA analysis)
# Isolation type: user

set -e

echo "========================================"
echo "CONTROLLED VARIATION LAMBDA SWEEP"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo ""

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Configuration
ISOLATION_TYPE="user"
LAYER=30
NUM_EPOCHS=20
OUTPUT_DIR="outputs/probes/controlled_variation/orthogonal"

echo "Configuration:"
echo "  Isolation type: $ISOLATION_TYPE"
echo "  Layer: $LAYER"
echo "  Epochs: $NUM_EPOCHS"
echo "  Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Lambda values to test
LAMBDAS=(0 10 100 1000)

for LAMBDA in "${LAMBDAS[@]}"; do
    echo ""
    echo "========================================"
    echo "Training with Lambda=$LAMBDA"
    echo "========================================"
    echo ""

    python3 probes/scripts/training/train_controlled_variation_orthogonal_probe.py \
        --isolation-type $ISOLATION_TYPE \
        --layer $LAYER \
        --lambda-reg $LAMBDA \
        --num-epochs $NUM_EPOCHS \
        --batch-size 32 \
        --lr 1e-3 \
        --weight-decay 10.0 \
        --output $OUTPUT_DIR \
        --seed 42

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Training failed for lambda=$LAMBDA"
        exit 1
    fi

    echo ""
    echo "✓ Lambda=$LAMBDA completed successfully"
    echo ""

    # Print results
    RESULT_DIR="${OUTPUT_DIR}/${ISOLATION_TYPE}_layer${LAYER}_lambda${LAMBDA}"
    if [ -f "${RESULT_DIR}/results.json" ]; then
        echo "Results for lambda=$LAMBDA:"
        python3 -c "
import json
with open('${RESULT_DIR}/results.json', 'r') as f:
    results = json.load(f)
print(f\"  Validation accuracy: {results['best_val_acc']:.4f}\")
print(f\"  Test accuracy (in-dist): {results['test_results']['accuracy']:.4f}\")
if results.get('cross_isolation_results'):
    print(f\"  Cross-isolation accuracy: {results['cross_isolation_results']['accuracy']:.4f}\")
    specificity = 1 - results['cross_isolation_results']['accuracy']
    print(f\"  Source specificity: {specificity:.4f}\")
"
        echo ""
    fi
done

echo ""
echo "========================================"
echo "LAMBDA SWEEP COMPLETE"
echo "========================================"
echo "Completed at: $(date)"
echo ""

# Summary comparison
echo "Summary across all lambda values:"
echo ""
echo "Lambda | Val Acc | Test Acc | Cross Acc | Specificity"
echo "-------|---------|----------|-----------|------------"

for LAMBDA in "${LAMBDAS[@]}"; do
    RESULT_DIR="${OUTPUT_DIR}/${ISOLATION_TYPE}_layer${LAYER}_lambda${LAMBDA}"
    if [ -f "${RESULT_DIR}/results.json" ]; then
        python3 -c "
import json
with open('${RESULT_DIR}/results.json', 'r') as f:
    results = json.load(f)
val_acc = results['best_val_acc']
test_acc = results['test_results']['accuracy']
cross_acc = results.get('cross_isolation_results', {}).get('accuracy', 0.0) if results.get('cross_isolation_results') else 0.0
specificity = 1 - cross_acc if cross_acc > 0 else 0.0
print(f\"{$LAMBDA:6} | {val_acc:7.4f} | {test_acc:8.4f} | {cross_acc:9.4f} | {specificity:11.4f}\")
"
    fi
done

echo ""
echo "✓ All training runs completed successfully!"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""

exit 0
