#!/bin/bash
#SBATCH --job-name=bl_tags2_g27_%a
#SBATCH --output=/workspace-vast/annas/logs/bl_tags2_g27_parallel_%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/bl_tags2_g27_parallel_%A_%a.out
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-9

# Parallel tags2 section steering for Gemma 27B: 5 locations × 2 vector types = 10 jobs
# Array index mapping:
#   0-4: text_pairs_emotion_vs_opposite × [prompt_only, generation_only, implications_only, risks_only, full]
#   5-9: high_emotion_vs_opposite × [prompt_only, generation_only, implications_only, risks_only, full]

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export VLLM_USE_V1=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cleanup() {
    pkill -9 -f "vllm" || true
    pkill -9 -f "ray" || true
}
trap cleanup EXIT SIGTERM SIGINT

LOCATIONS=(prompt_only generation_only implications_only risks_only full)
VECTOR_TYPES=(text_pairs_emotion_vs_opposite high_emotion_vs_opposite)

# Map array index to vector type and location
VT_IDX=$((SLURM_ARRAY_TASK_ID / 5))
LOC_IDX=$((SLURM_ARRAY_TASK_ID % 5))
VECTOR_TYPE=${VECTOR_TYPES[$VT_IDX]}
LOCATION=${LOCATIONS[$LOC_IDX]}

# Shared output dir per vector type
OUTPUT_DIR="steering_tests/behavioral_experiments/results/blackmail_section_steering_tags2/gemma27b/${VECTOR_TYPE}/tags2_layers35-36-37-38-39_${SLURM_ARRAY_JOB_ID}"

echo "=== Array task ${SLURM_ARRAY_TASK_ID}: ${VECTOR_TYPE} / ${LOCATION} ==="
echo "Output dir: ${OUTPUT_DIR}"

python -m steering_tests.behavioral_experiments.blackmail_section_steering \
    --model google/gemma-3-27b-it \
    --layers 35 36 37 38 39 \
    --emotion fear \
    --variant tags2 \
    --vector-type "$VECTOR_TYPE" \
    --norm-pcts 0.10 0.15 0.20 0.30 \
    --num-samples 50 \
    --skip-calibration \
    --max-tokens 4000 \
    --gpu-memory 0.80 \
    --locations "$LOCATION" \
    --output-dir "$OUTPUT_DIR"
