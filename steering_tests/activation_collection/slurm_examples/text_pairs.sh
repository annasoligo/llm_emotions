#!/bin/bash
#SBATCH --job-name=acts_text_pairs
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/text_pairs_%j.out
#SBATCH --error=logs/text_pairs_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Example: Collect activations for emotion text pairs dataset

# Configuration
MODEL="google/gemma-2-9b-it"
INPUT="steering_tests/data/emotion_text_pairs_24_full_500.jsonl"
OUTPUT="steering_tests/activations/emotion_pairs_gemma2_9b"
LAYERS="all"

echo "====================================================="
echo "Text Pairs Activation Collection"
echo "====================================================="
echo "Model: $MODEL"
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo "Layers: $LAYERS"
echo "====================================================="

# Create output directory
mkdir -p steering_tests/activations
mkdir -p logs

# Run collection (text mode - no chat template)
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
