#!/bin/bash
#SBATCH --job-name=mistral_high
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=/workspace-vast/annas/logs/mistral_nemo_high_%j.out
#SBATCH --error=/workspace-vast/annas/logs/mistral_nemo_high_%j.out

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

MODEL="mistralai/Mistral-Nemo-Instruct-2407"
INPUT="steering_tests/data/emotion_prompts_MODEL_500_HIGH.jsonl"
OUTPUT="steering_tests/activations/mistral_nemo_high_emotion_prompts"

echo "====================================================="
echo "Mistral Nemo High Emotion Prompts Collection"
echo "Model: $MODEL"
echo "====================================================="

mkdir -p steering_tests/activations

python steering_tests/activation_collection/collect.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --mode chat \
  --layers all \
  --dtype bfloat16 \
  --batch-size 16 \
  --batch-save 100 \
  --resume

echo "DONE!"
