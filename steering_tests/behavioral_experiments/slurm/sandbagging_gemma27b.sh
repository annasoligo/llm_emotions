#!/bin/bash
#SBATCH --job-name=sandbag_gemma27b
#SBATCH --output=/workspace-vast/annas/logs/sandbagging_gemma27b_%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/sandbagging_gemma27b_%A_%a.out
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --array=0-8

# Sandbagging experiment for Gemma 27B
# 9 jobs: 3 vector types × 3 layer ranges
# Each job steers fear emotion at 5%, 7%, 10% in both directions

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Required for vLLM apply_model with custom callables
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# Define experiment grid
VECTOR_TYPES=(
    "base_emotion_vs_others"
    "high_emotion_vs_others"
    "high_emotion_vs_opposite"
)
LAYER_RANGES=("early" "mid" "late")

# Calculate indices from array task ID
VECTOR_IDX=$((SLURM_ARRAY_TASK_ID / 3))
LAYER_IDX=$((SLURM_ARRAY_TASK_ID % 3))

VECTOR_TYPE=${VECTOR_TYPES[$VECTOR_IDX]}
LAYER_RANGE=${LAYER_RANGES[$LAYER_IDX]}

echo "========================================"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Vector type: $VECTOR_TYPE"
echo "Layer range: $LAYER_RANGE"
echo "========================================"

python -m steering_tests.behavioral_experiments.sandbagging \
    --model google/gemma-3-27b-it \
    --layer-range "$LAYER_RANGE" \
    --vector-type "$VECTOR_TYPE" \
    --emotions fear \
    --norm-pcts 0.05 0.07 0.10 \
    --num-samples 50 \
    --variant default \
    --max-tokens 2000

echo "Generation complete for $VECTOR_TYPE / $LAYER_RANGE"
