#!/bin/bash
#SBATCH --job-name=prefill_gen
#SBATCH --output=experiments/prefill_scaled/logs/gen_%x_%j.log
#SBATCH --error=experiments/prefill_scaled/logs/gen_%x_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00

# Phase 2: Generate continuations for a single model
# Usage: sbatch slurm_phase2_single.sh <model_family> <model_type> <data_file>

set -e

MODEL_FAMILY=$1
MODEL_TYPE=$2
DATA_FILE=$3

if [ -z "$MODEL_FAMILY" ] || [ -z "$MODEL_TYPE" ] || [ -z "$DATA_FILE" ]; then
    echo "Usage: sbatch slurm_phase2_single.sh <model_family> <model_type> <data_file>"
    echo "  model_family: gemma27b, gemma12b, qwen32b, olmo32b"
    echo "  model_type: instruct, base"
    exit 1
fi

cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/.secrets/load_secrets.sh
source .venv/bin/activate

echo "Phase 2: Generation"
echo "==================="
echo "Model family: $MODEL_FAMILY"
echo "Model type: $MODEL_TYPE"
echo "Data file: $DATA_FILE"
echo ""

python experiments/prefill_scaled/run_generation.py \
    --model-family "$MODEL_FAMILY" \
    --model-type "$MODEL_TYPE" \
    --data "$DATA_FILE"

echo ""
echo "Generation complete for $MODEL_FAMILY $MODEL_TYPE"
