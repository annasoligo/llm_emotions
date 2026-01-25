#!/bin/bash
#SBATCH --job-name=qwen14b_high
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=logs/qwen14b_high_%j.out
#SBATCH --error=logs/qwen14b_high_%j.err

source .venv/bin/activate
source /workspace-vast/annas/.secrets/load_secrets.sh

MODEL="Qwen/Qwen3-14B"
INPUT="steering_tests/data/emotion_prompts_MODEL_500_HIGH.jsonl"
OUTPUT="steering_tests/activations/test_qwen14b_high"

echo "====================================================="
echo "Qwen-14B High Emotion Prompts Collection"
echo "====================================================="

mkdir -p steering_tests/activations logs

python steering_tests/activation_collection/collect.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --mode chat \
  --layers all \
  --dtype bfloat16 \
  --batch-save 100 \
  --resume

echo "DONE!"
