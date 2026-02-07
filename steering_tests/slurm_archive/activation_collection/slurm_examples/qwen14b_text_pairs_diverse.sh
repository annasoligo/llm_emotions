#!/bin/bash
#SBATCH --job-name=qwen14b_tp_diverse
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=/workspace-vast/annas/logs/qwen14b_tp_diverse_%j.out
#SBATCH --error=/workspace-vast/annas/logs/qwen14b_tp_diverse_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
export PYTHONPATH=/workspace-vast/annas/git/research-tools:$PYTHONPATH

MODEL="Qwen/Qwen3-14B"
INPUT="steering_tests/data/emotion_text_pairs_24_full_500.jsonl"
OUTPUT="steering_tests/activations/qwen14b_text_pairs"

echo "====================================================="
echo "Qwen-14B Text Pairs (diverse topics) Collection"
echo "Model: $MODEL"
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo "Start: $(date)"
echo "====================================================="

python steering_tests/activation_collection/collect.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --mode text \
  --layers all \
  --start-token 20 \
  --dtype bfloat16 \
  --batch-save 50 \
  --resume

echo "DONE: $(date)"
