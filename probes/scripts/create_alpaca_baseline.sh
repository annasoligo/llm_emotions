#!/bin/bash
# Create new baseline activations from Alpaca dataset using OpenRouter + local model
#
# Usage: ./create_alpaca_baseline.sh

set -e

echo "=============================================================================="
echo "CREATING ALPACA BASELINE ACTIVATIONS"
echo "=============================================================================="
echo ""

# Check for OpenRouter API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "ERROR: OPENROUTER_API_KEY environment variable not set"
    echo "Get your key from https://openrouter.ai/keys"
    echo ""
    echo "Then run:"
    echo "  export OPENROUTER_API_KEY='your-key-here'"
    exit 1
fi

# Configuration
NUM_SAMPLES=512
MAX_CONCURRENT=20
OUTPUT_JSONL="data/alpaca_responses.jsonl"
OUTPUT_BASELINE="data/baselines/alpaca_gemma27b"
MODEL_NAME="unsloth/gemma-3-27b-it"
LAYERS="0-62"

echo "Configuration:"
echo "  Samples: $NUM_SAMPLES"
echo "  Concurrent requests: $MAX_CONCURRENT"
echo "  Model: $MODEL_NAME"
echo "  Layers: $LAYERS"
echo "  Output: $OUTPUT_BASELINE"
echo ""

# Step 1: Generate responses via OpenRouter
echo "=============================================================================="
echo "STEP 1: Generating responses via OpenRouter"
echo "=============================================================================="
echo ""

python probes/scripts/generate_alpaca_responses.py \
    --num-samples $NUM_SAMPLES \
    --max-concurrent $MAX_CONCURRENT \
    --output $OUTPUT_JSONL \
    --seed 42

echo ""
echo "✓ Step 1 complete"
echo ""

# Step 2: Collect baseline activations
echo "=============================================================================="
echo "STEP 2: Collecting baseline activations from local model"
echo "=============================================================================="
echo ""

python probes/scripts/compute_baseline_activations_from_jsonl.py \
    --input $OUTPUT_JSONL \
    --output $OUTPUT_BASELINE \
    --model-name $MODEL_NAME \
    --layers $LAYERS \
    --device cuda

echo ""
echo "✓ Step 2 complete"
echo ""

# Summary
echo "=============================================================================="
echo "SUCCESS! NEW BASELINE CREATED"
echo "=============================================================================="
echo ""
echo "Baseline location: $OUTPUT_BASELINE"
echo ""
echo "To use this baseline in your experiments:"
echo ""
echo "  from probes.scripts.wildchat_baseline_loader import WildChatBaselineLoader"
echo "  from pathlib import Path"
echo ""
echo "  loader = WildChatBaselineLoader("
echo "      aggregation_type='assistant_turn',"
echo "      baseline_dir=Path('$OUTPUT_BASELINE')"
echo "  )"
echo ""
echo "Or in token_level_experiment_v2.py, update:"
echo "  USE_WILDCHAT_NORMALIZATION = True"
echo "  WILDCHAT_AGGREGATION = 'assistant_turn'"
echo ""
echo "And modify TokenLevelExperiment to use the new baseline_dir."
echo ""
