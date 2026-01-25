#!/bin/bash
#SBATCH --job-name=acts_appraisal
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/appraisal_%j.out
#SBATCH --error=logs/appraisal_%j.err

# Example: Collect activations for appraisal minimal pairs

# Activate virtual environment
source .venv/bin/activate

# Load secrets
source /workspace-vast/annas/.secrets/load_secrets.sh

# Configuration
MODEL="google/gemma-2-9b-it"
DATA_DIR="steering_tests/data/appraisal_minimal_pairs"
OUTPUT_DIR="steering_tests/activations/appraisal"
LAYERS="all"

echo "====================================================="
echo "Appraisal Activation Collection"
echo "====================================================="
echo "Model: $MODEL"
echo "Data dir: $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Layers: $LAYERS"
echo "====================================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Run collection
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
