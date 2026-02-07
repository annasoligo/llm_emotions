#!/bin/bash
#SBATCH --job-name=llama70b_base
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --mem=320G
#SBATCH --time=8:00:00
#SBATCH --output=/workspace-vast/annas/logs/llama70b_base_%j.out
#SBATCH --error=/workspace-vast/annas/logs/llama70b_base_%j.out

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

MODEL="meta-llama/Llama-3.3-70B-Instruct"
INPUT="steering_tests/data/emotion_prompts_MODEL_500.jsonl"
OUTPUT="steering_tests/activations/llama70b_emotion_prompts"

echo "====================================================="
echo "Llama 3.3 70B Base Emotion Prompts Collection"
echo "Model: $MODEL"
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo "====================================================="

mkdir -p steering_tests/activations

python steering_tests/activation_collection/collect.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --mode chat \
  --layers all \
  --dtype bfloat16 \
  --batch-size 8 \
  --batch-save 50 \
  --resume

echo "DONE!"
