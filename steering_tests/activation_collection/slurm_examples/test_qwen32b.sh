#!/bin/bash
#SBATCH --job-name=test_qwen32b
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/test_qwen32b_%j.out
#SBATCH --error=logs/test_qwen32b_%j.err

# Test: Collect activations on Qwen-32B model

# Activate virtual environment
source .venv/bin/activate

# Load secrets
source /workspace-vast/annas/.secrets/load_secrets.sh

# Configuration
MODEL="Qwen/Qwen2.5-32B-Instruct"
INPUT="steering_tests/data/emotion_prompts_MODEL_500.jsonl"
OUTPUT="steering_tests/activations/test_qwen32b"
LAYERS="all"

echo "====================================================="
echo "Testing Qwen-32B Activation Collection"
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
