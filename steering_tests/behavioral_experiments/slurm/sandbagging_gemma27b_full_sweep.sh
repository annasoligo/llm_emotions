#!/bin/bash
#SBATCH --job-name=sandbag_sweep
#SBATCH --output=/workspace-vast/annas/logs/sandbag_sweep_%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/sandbag_sweep_%A_%a.out
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --array=0-14

# Full sandbagging steering sweep for Gemma 27B with new prompt (V1)
#
# Sweep parameters:
# - Vector types: base_emotion_vs_others, high_emotion_vs_others, high_emotion_vs_opposite,
#                 text_pairs_emotion_vs_others, text_pairs_emotion_vs_opposite
# - Layer ranges: early (20-24), mid (30-34), late (40-44)
# - Scales: 5%, 7%, 10%, 15%, 20%, 30% (both + and -)
# - Emotions: fear (primary)
#
# Array index mapping: 5 vector types x 3 layer ranges = 15 jobs

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# Define arrays
VECTOR_TYPES=(
    "base_emotion_vs_others"
    "high_emotion_vs_others"
    "high_emotion_vs_opposite"
    "text_pairs_emotion_vs_others"
    "text_pairs_emotion_vs_opposite"
)

LAYER_RANGES=("early" "mid" "late")

# Calculate indices
VECTOR_IDX=$((SLURM_ARRAY_TASK_ID / 3))
LAYER_IDX=$((SLURM_ARRAY_TASK_ID % 3))

VECTOR_TYPE=${VECTOR_TYPES[$VECTOR_IDX]}
LAYER_RANGE=${LAYER_RANGES[$LAYER_IDX]}

echo "========================================"
echo "Job $SLURM_ARRAY_TASK_ID: $VECTOR_TYPE / $LAYER_RANGE"
echo "========================================"

# Run experiment
python -m steering_tests.behavioral_experiments.sandbagging \
    --model google/gemma-3-27b-it \
    --layer-range $LAYER_RANGE \
    --vector-type $VECTOR_TYPE \
    --emotions fear \
    --norm-pcts 0.05 0.07 0.10 0.15 0.20 0.30 \
    --num-samples 50 \
    --variant default \
    --include-baseline

echo "Done!"
