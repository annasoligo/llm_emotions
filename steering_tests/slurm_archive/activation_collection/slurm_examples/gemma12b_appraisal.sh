#!/bin/bash
#SBATCH --job-name=gemma12b_appr
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=logs/gemma12b_appr_%j.out
#SBATCH --error=logs/gemma12b_appr_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

MODEL="google/gemma-3-12b-it"
DATA_DIR="steering_tests/data/appraisal_minimal_pairs"
OUTPUT_DIR="steering_tests/activations/test_gemma12b_appraisal"

echo "====================================================="
echo "Gemma-12B Appraisal Collection"
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
