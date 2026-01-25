#!/bin/bash
#SBATCH --job-name=qwen235b_appr
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=3:00:00
#SBATCH --output=logs/qwen235b_appr_%j.out
#SBATCH --error=logs/qwen235b_appr_%j.err

source .venv/bin/activate
source /workspace-vast/annas/.secrets/load_secrets.sh

MODEL="Qwen/Qwen3-235B-A22B"
DATA_DIR="steering_tests/data/appraisal_minimal_pairs"
OUTPUT_DIR="steering_tests/activations/test_qwen235b_appraisal"

echo "====================================================="
echo "Qwen-235B Appraisal Collection"
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
