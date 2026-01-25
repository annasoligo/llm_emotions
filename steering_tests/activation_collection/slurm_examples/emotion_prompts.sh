#!/bin/bash
#SBATCH --job-name=acts_emotion_prompts
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/emotion_prompts_%j.out
#SBATCH --error=logs/emotion_prompts_%j.err

# Example: Collect activations for emotion prompts dataset

# Activate virtual environment
source .venv/bin/activate

# Load secrets (API keys, etc.)
source /workspace-vast/annas/.secrets/load_secrets.sh

# Configuration
MODEL="google/gemma-2-9b-it"
INPUT="steering_tests/data/emotion_prompts_MODEL_500.jsonl"
OUTPUT="steering_tests/activations/emotion_prompts_gemma2_9b"
LAYERS="all"

echo "====================================================="
echo "Emotion Prompts Activation Collection"
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
