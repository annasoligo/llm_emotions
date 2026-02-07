#!/bin/bash
#SBATCH --job-name=test_gemma3_27b
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=logs/test_gemma3_27b_%j.out
#SBATCH --error=logs/test_gemma3_27b_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Test: Collect activations on Gemma-3-27B model

# Configuration
MODEL="google/gemma-3-27b-it"
INPUT="steering_tests/data/emotion_prompts_MODEL_500.jsonl"
OUTPUT="steering_tests/activations/test_gemma3_27b"
LAYERS="all"

echo "====================================================="
echo "Testing Gemma-3-27B Activation Collection"
echo "====================================================="
echo "Model: $MODEL"
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo "Layers: $LAYERS"
echo "====================================================="

# Create output directory
mkdir -p steering_tests/activations
mkdir -p logs

# Run collection
python steering_tests/activation_collection/collect.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --mode chat \
  --layers "$LAYERS" \
  --dtype bfloat16 \
  --batch-save 100 \
  --resume

echo "====================================================="
echo "DONE!"
echo "====================================================="
