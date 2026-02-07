#!/bin/bash
#SBATCH --job-name=gemma27b_tp_diverse
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=/workspace-vast/annas/logs/gemma27b_tp_diverse_%j.out
#SBATCH --error=/workspace-vast/annas/logs/gemma27b_tp_diverse_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
export PYTHONPATH=/workspace-vast/annas/git/research-tools:$PYTHONPATH

# Collect activations from the new diverse-topic text pairs data
MODEL="google/gemma-3-27b-it"
INPUT="steering_tests/data/emotion_text_pairs_24_full_500.jsonl"
OUTPUT="steering_tests/activations/gemma3_27b_text_pairs"
LAYERS="all"

echo "====================================================="
echo "Gemma-3-27B Text Pairs (diverse topics) Collection"
echo "====================================================="
echo "Model: $MODEL"
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo "Start: $(date)"
echo "====================================================="

mkdir -p steering_tests/activations
mkdir -p /workspace-vast/annas/logs

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
echo "DONE: $(date)"
echo "====================================================="
