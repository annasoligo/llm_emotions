#!/bin/bash
#SBATCH --job-name=qwen32b_high
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/qwen32b_high_%j.out
#SBATCH --error=logs/qwen32b_high_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Configuration
MODEL="Qwen/Qwen3-32B"
INPUT="steering_tests/data/emotion_prompts_MODEL_500_HIGH.jsonl"
OUTPUT="steering_tests/activations/test_qwen32b_high"
LAYERS="all"

echo "====================================================="
echo "Qwen-32B High Emotion Prompts Collection"
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
