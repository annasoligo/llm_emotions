#!/bin/bash
#SBATCH --job-name=qwen235b_high
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=6:00:00
#SBATCH --output=logs/qwen235b_high_%j.out
#SBATCH --error=logs/qwen235b_high_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

MODEL="Qwen/Qwen3-235B-A22B"
INPUT="steering_tests/data/emotion_prompts_MODEL_500_HIGH.jsonl"
OUTPUT="steering_tests/activations/test_qwen235b_high"

echo "====================================================="
echo "Qwen-235B High Emotion Prompts Collection"
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
