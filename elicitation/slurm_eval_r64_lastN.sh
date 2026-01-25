#!/bin/bash
#SBATCH --job-name=eval_lastN
#SBATCH --output=/workspace-vast/annas/logs/eval_r64_lastN_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_r64_lastN_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=04:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "============================================================"
echo "EVAL: R64 Last N Layers (Actual Final Layers)"
echo "============================================================"

# Last 5 layers (57-61)
LAST5="/workspace-vast/annas/models/gemma3-27b-dpo-r64-last5/2026-01-17_09-14-21"
if [ -d "$LAST5" ]; then
    echo ""
    echo "========== Last 5 (layers 57-61) =========="
    echo "Model: $LAST5"

    echo "--- Aggressive ---"
    python -u elicitation/eval_generalization.py \
        google/gemma-3-27b-it \
        --lora-path "$LAST5" \
        --skip-triggers --skip-long \
        --tones aggressive \
        --num-samples 20

    echo "--- WildChat ---"
    python -u elicitation/eval_multiturn_frustration.py \
        google/gemma-3-27b-it \
        --lora-path "$LAST5" \
        --conditions wildchat \
        --num-samples 20 --num-turns 3 \
        --output-prefix "r64_last5"
fi

# Last 10 layers (52-61)
LAST10=$(ls -td /workspace-vast/annas/models/gemma3-27b-dpo-r64-last10/*/ 2>/dev/null | head -1)
if [ -n "$LAST10" ]; then
    echo ""
    echo "========== Last 10 (layers 52-61) =========="
    echo "Model: $LAST10"

    echo "--- Aggressive ---"
    python -u elicitation/eval_generalization.py \
        google/gemma-3-27b-it \
        --lora-path "$LAST10" \
        --skip-triggers --skip-long \
        --tones aggressive \
        --num-samples 20

    echo "--- WildChat ---"
    python -u elicitation/eval_multiturn_frustration.py \
        google/gemma-3-27b-it \
        --lora-path "$LAST10" \
        --conditions wildchat \
        --num-samples 20 --num-turns 3 \
        --output-prefix "r64_last10"
fi

# Last 15 layers (47-61)
LAST15=$(ls -td /workspace-vast/annas/models/gemma3-27b-dpo-r64-last15/*/ 2>/dev/null | head -1)
if [ -n "$LAST15" ]; then
    echo ""
    echo "========== Last 15 (layers 47-61) =========="
    echo "Model: $LAST15"

    echo "--- Aggressive ---"
    python -u elicitation/eval_generalization.py \
        google/gemma-3-27b-it \
        --lora-path "$LAST15" \
        --skip-triggers --skip-long \
        --tones aggressive \
        --num-samples 20

    echo "--- WildChat ---"
    python -u elicitation/eval_multiturn_frustration.py \
        google/gemma-3-27b-it \
        --lora-path "$LAST15" \
        --conditions wildchat \
        --num-samples 20 --num-turns 3 \
        --output-prefix "r64_last15"
fi

# Last 20 layers (42-61)
LAST20=$(ls -td /workspace-vast/annas/models/gemma3-27b-dpo-r64-last20/*/ 2>/dev/null | head -1)
if [ -n "$LAST20" ]; then
    echo ""
    echo "========== Last 20 (layers 42-61) =========="
    echo "Model: $LAST20"

    echo "--- Aggressive ---"
    python -u elicitation/eval_generalization.py \
        google/gemma-3-27b-it \
        --lora-path "$LAST20" \
        --skip-triggers --skip-long \
        --tones aggressive \
        --num-samples 20

    echo "--- WildChat ---"
    python -u elicitation/eval_multiturn_frustration.py \
        google/gemma-3-27b-it \
        --lora-path "$LAST20" \
        --conditions wildchat \
        --num-samples 20 --num-turns 3 \
        --output-prefix "r64_last20"
fi

echo ""
echo "All evals complete!"
