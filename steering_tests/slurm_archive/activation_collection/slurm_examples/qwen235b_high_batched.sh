#!/bin/bash
#SBATCH --job-name=qwen235b_high_batched
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/qwen235b_high_batched_%j.out
#SBATCH --error=logs/qwen235b_high_batched_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

MODEL="Qwen/Qwen3-235B-A22B"
INPUT="steering_tests/data/emotion_prompts_MODEL_500_HIGH.jsonl"
OUTPUT="steering_tests/activations/qwen235b_high_emotion_prompts"

echo "====================================================="
echo "Qwen-235B High Emotion Prompts Collection (BATCHED)"
echo "====================================================="
echo "Model: $MODEL"
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo "====================================================="

python steering_tests/activation_collection/collect.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --mode chat \
  --layers all \
  --dtype bfloat16 \
  --batch-size 8 \
  --batch-save 100 \
  --resume

echo "Collection complete!"
