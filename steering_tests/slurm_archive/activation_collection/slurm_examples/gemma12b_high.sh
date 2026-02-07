#!/bin/bash
#SBATCH --job-name=gemma12b_high
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=5:00:00
#SBATCH --output=logs/gemma12b_high_%j.out
#SBATCH --error=logs/gemma12b_high_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

MODEL="google/gemma-3-12b-it"
INPUT="steering_tests/data/emotion_prompts_MODEL_500_HIGH.jsonl"
OUTPUT="steering_tests/activations/test_gemma12b_high"

echo "====================================================="
echo "Gemma-12B High Emotion Prompts Collection"
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
