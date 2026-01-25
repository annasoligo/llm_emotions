#!/bin/bash
#SBATCH --job-name=gemma12b_emo
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=5:00:00
#SBATCH --output=logs/gemma12b_emo_%j.out
#SBATCH --error=logs/gemma12b_emo_%j.err

source .venv/bin/activate
source /workspace-vast/annas/.secrets/load_secrets.sh

MODEL="google/gemma-3-12b-it"
INPUT="steering_tests/data/emotion_prompts_MODEL_500.jsonl"
OUTPUT="steering_tests/activations/test_gemma12b"

echo "====================================================="
echo "Gemma-12B Emotion Prompts Collection"
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
