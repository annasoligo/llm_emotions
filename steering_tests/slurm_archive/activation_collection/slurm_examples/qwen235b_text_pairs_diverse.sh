#!/bin/bash
#SBATCH --job-name=qwen235b_tp_diverse
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --output=/workspace-vast/annas/logs/qwen235b_tp_diverse_%j.out
#SBATCH --error=/workspace-vast/annas/logs/qwen235b_tp_diverse_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
export PYTHONPATH=/workspace-vast/annas/git/research-tools:$PYTHONPATH

MODEL="Qwen/Qwen3-235B-A22B"
INPUT="steering_tests/data/emotion_text_pairs_24_full_500.jsonl"
OUTPUT="steering_tests/activations/qwen235b_text_pairs"

echo "====================================================="
echo "Qwen-235B Text Pairs (diverse topics) Collection"
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
