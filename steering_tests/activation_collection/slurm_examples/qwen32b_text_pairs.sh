#!/bin/bash
#SBATCH --job-name=qwen32b_pairs
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=logs/qwen32b_pairs_%j.out
#SBATCH --error=logs/qwen32b_pairs_%j.err

# Activate virtual environment
source .venv/bin/activate

# Load secrets
source /workspace-vast/annas/.secrets/load_secrets.sh

# Configuration
MODEL="Qwen/Qwen2.5-32B-Instruct"
INPUT="steering_tests/data/emotion_text_pairs_24_full_500.jsonl"
OUTPUT="steering_tests/activations/test_qwen32b_text_pairs"
LAYERS="all"

echo "====================================================="
echo "Qwen-32B Text Pairs Collection"
echo "====================================================="
echo "Model: $MODEL"
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo "====================================================="

mkdir -p steering_tests/activations
mkdir -p logs

python steering_tests/activation_collection/collect.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --mode text \
  --layers "$LAYERS" \
  --start-token 20 \
  --dtype bfloat16 \
  --batch-save 50 \
  --resume

echo "====================================================="
echo "DONE!"
echo "====================================================="
