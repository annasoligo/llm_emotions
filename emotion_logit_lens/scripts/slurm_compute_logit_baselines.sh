#!/bin/bash
#SBATCH --job-name=logit_baselines
#SBATCH --output=/workspace-vast/annas/git/research-tools/emotion_logit_lens/logs/logit_baselines_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/emotion_logit_lens/logs/logit_baselines_%A.err
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
uv pip show transformers > /dev/null 2>&1 || uv pip install transformers
uv pip show torch > /dev/null 2>&1 || uv pip install torch
uv pip show tqdm > /dev/null 2>&1 || uv pip install tqdm

echo "=============================================================================="
echo "COMPUTING LOGIT BASELINES FOR EMOTION DETECTION"
echo "=============================================================================="
echo ""

# Configuration
MODEL_NAME="google/gemma-3-27b-it"
ACTIVATION_BASELINE_DIR="data/baselines/alpaca_gemma27b_v2/google_gemma_3_27b_it"
OUTPUT_DIR="data/baselines/logit_emotion_alpaca/google_gemma_3_27b_it"
LAYERS="0-61"
BATCH_SIZE=32

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Activation baseline dir: $ACTIVATION_BASELINE_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Layers: $LAYERS"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Run baseline computation
python emotion_logit_lens/scripts/compute_logit_baselines.py \
    --model $MODEL_NAME \
    --activation-baseline-dir $ACTIVATION_BASELINE_DIR \
    --output-dir $OUTPUT_DIR \
    --layers $LAYERS \
    --batch-size $BATCH_SIZE

echo ""
echo "=============================================================================="
echo "SUCCESS! LOGIT BASELINES CREATED"
echo "=============================================================================="
echo ""
echo "Baseline location: $OUTPUT_DIR"
echo ""
echo "Files created:"
ls -lh $OUTPUT_DIR/ | head -10
echo ""
echo "Total files:"
ls $OUTPUT_DIR/*.json | wc -l
echo ""
