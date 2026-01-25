#!/bin/bash
#SBATCH --job-name=qwen14b_appr
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=logs/qwen14b_appr_%j.out
#SBATCH --error=logs/qwen14b_appr_%j.err

source .venv/bin/activate
source /workspace-vast/annas/.secrets/load_secrets.sh

MODEL="Qwen/Qwen3-14B"
DATA_DIR="steering_tests/data/appraisal_minimal_pairs"
OUTPUT_DIR="steering_tests/activations/test_qwen14b_appraisal"

echo "====================================================="
echo "Qwen-14B Appraisal Collection"
echo "====================================================="

mkdir -p "$OUTPUT_DIR" logs

python steering_tests/activation_collection/collect_appraisal.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --model "$MODEL" \
  --layers all \
  --dtype bfloat16 \
  --resume

echo "DONE!"
