#!/bin/bash
#SBATCH --job-name=qwen14b_pairs
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=logs/qwen14b_pairs_%j.out
#SBATCH --error=logs/qwen14b_pairs_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

MODEL="Qwen/Qwen3-14B"
INPUT="steering_tests/data/emotion_text_pairs_24_full_500.jsonl"
OUTPUT="steering_tests/activations/test_qwen14b_text_pairs"

echo "====================================================="
echo "Qwen-14B Text Pairs Collection"
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
