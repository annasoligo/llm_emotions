#!/bin/bash

# Evaluate all lambda values on conversation data

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

LAMBDAS=(0.0 10.0 100.0 1000.0)

echo "========================================"
echo "EVALUATING ALL LAMBDA VALUES ON CONVERSATION DATA"
echo "========================================"
echo ""

for LAMBDA in "${LAMBDAS[@]}"; do
    echo ""
    echo "========================================"
    echo "Lambda=$LAMBDA"
    echo "========================================"

    python3 probes/scripts/evaluation/eval_controlled_variation_on_conversations.py \
        --probe-path outputs/probes/controlled_variation/orthogonal/user_layer30_lambda${LAMBDA}/model.pt \
        --layer 30 \
        --emotion-type user \
        2>&1 | grep -A1 "^Accuracy:"

    echo ""
done

echo "========================================"
echo "SUMMARY"
echo "========================================"
echo ""
echo "Lambda   | Accuracy"
echo "---------|----------"

for LAMBDA in "${LAMBDAS[@]}"; do
    ACC=$(python3 probes/scripts/evaluation/eval_controlled_variation_on_conversations.py \
        --probe-path outputs/probes/controlled_variation/orthogonal/user_layer30_lambda${LAMBDA}/model.pt \
        --layer 30 \
        --emotion-type user \
        2>&1 | grep "^Accuracy:" | awk '{print $2}')
    printf "%8s | %s\n" "$LAMBDA" "$ACC"
done
