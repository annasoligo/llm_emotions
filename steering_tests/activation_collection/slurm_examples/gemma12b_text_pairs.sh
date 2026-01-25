#!/bin/bash
#SBATCH --job-name=gemma12b_pairs
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=logs/gemma12b_pairs_%j.out
#SBATCH --error=logs/gemma12b_pairs_%j.err

source .venv/bin/activate
source /workspace-vast/annas/.secrets/load_secrets.sh

MODEL="google/gemma-3-12b-it"
INPUT="steering_tests/data/emotion_text_pairs_24_full_500.jsonl"
OUTPUT="steering_tests/activations/test_gemma12b_text_pairs"

echo "====================================================="
echo "Gemma-12B Text Pairs Collection"
echo "====================================================="

mkdir -p steering_tests/activations logs

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

echo "DONE!"
