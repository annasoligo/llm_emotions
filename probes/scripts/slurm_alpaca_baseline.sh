#!/bin/bash
#SBATCH --job-name=alpaca_baseline
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/alpaca_baseline_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/alpaca_baseline_%A.err
#SBATCH --time=02:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

set -e

# Load environment
cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate venv
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Activated venv"
else
    echo "⚠ No venv found at .venv/bin/activate"
fi

# Set HuggingFace cache
export HF_HOME=/workspace-vast/pretrained_ckpts

# Install dependencies if needed (using uv pip for uv-managed venv)
uv pip show h5py > /dev/null 2>&1 || uv pip install h5py
uv pip show aiohttp > /dev/null 2>&1 || uv pip install aiohttp
uv pip show datasets > /dev/null 2>&1 || uv pip install datasets
uv pip show tqdm > /dev/null 2>&1 || uv pip install tqdm

# Check for OpenRouter API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set in secrets"
    exit 1
fi

echo "=============================================================================="
echo "CREATING ALPACA BASELINE ACTIVATIONS"
echo "=============================================================================="
echo ""

# Configuration
NUM_SAMPLES=${1:-512}
MAX_CONCURRENT=${2:-20}
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
echo "Files created:"
ls -lh $OUTPUT_BASELINE/ | head -20
echo ""
echo "To use this baseline, update your code to point to:"
echo "  baseline_dir=Path('$OUTPUT_BASELINE')"
echo ""
