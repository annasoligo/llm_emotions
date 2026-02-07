#!/bin/bash
#SBATCH --job-name=mistral_pairs
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=/workspace-vast/annas/logs/mistral_nemo_pairs_%j.out
#SBATCH --error=/workspace-vast/annas/logs/mistral_nemo_pairs_%j.out

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

MODEL="mistralai/Mistral-Nemo-Instruct-2407"
INPUT="steering_tests/data/emotion_text_pairs_24_full_500.jsonl"
OUTPUT="steering_tests/activations/mistral_nemo_text_pairs"

echo "====================================================="
echo "Mistral Nemo Text Pairs Collection"
echo "Model: $MODEL"
echo "====================================================="

mkdir -p steering_tests/activations

python steering_tests/activation_collection/collect.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --mode text \
  --layers all \
  --dtype bfloat16 \
  --batch-size 16 \
  --batch-save 100 \
  --start-token 20 \
  --resume

echo "DONE!"
