#!/bin/bash
#SBATCH --job-name=qwen32b_appr_scen
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=logs/qwen32b_appr_scenarios_%j.out
#SBATCH --error=logs/qwen32b_appr_scenarios_%j.err

# Activate virtual environment
source .venv/bin/activate

# Load secrets
source /workspace-vast/annas/.secrets/load_secrets.sh

# Configuration
MODEL="Qwen/Qwen2.5-32B-Instruct"
DATA_DIR="steering_tests/data/appraisal_minimal_pairs"
OUTPUT_DIR="steering_tests/activations/test_qwen32b_appraisal"
LAYERS="all"

echo "====================================================="
echo "Qwen-32B Appraisal Scenarios Collection"
echo "====================================================="
echo "Model: $MODEL"
echo "Data dir: $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "====================================================="

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

python steering_tests/activation_collection/collect_appraisal.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --model "$MODEL" \
  --layers "$LAYERS" \
  --dtype bfloat16 \
  --resume

echo "====================================================="
echo "DONE!"
echo "====================================================="
