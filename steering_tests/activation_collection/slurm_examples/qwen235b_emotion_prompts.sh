#!/bin/bash
#SBATCH --job-name=qwen235b_emo
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=6:00:00
#SBATCH --output=logs/qwen235b_emo_%j.out
#SBATCH --error=logs/qwen235b_emo_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Configuration
MODEL="Qwen/Qwen3-235B-A22B"
INPUT="steering_tests/data/emotion_prompts_MODEL_500.jsonl"
OUTPUT="steering_tests/activations/test_qwen235b"
LAYERS="all"

echo "====================================================="
echo "Qwen-235B Emotion Prompts Collection"
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
  --mode chat \
  --layers "$LAYERS" \
  --dtype bfloat16 \
  --batch-save 100 \
  --resume

echo "====================================================="
echo "DONE!"
echo "====================================================="
