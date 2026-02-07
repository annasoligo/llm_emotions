#!/bin/bash
#SBATCH --job-name=kl_hf_bh%a
#SBATCH --output=/workspace-vast/annas/git/research-tools/steering_tests/vector_testing/results/logs/%x_%j.out
#SBATCH --error=/workspace-vast/annas/git/research-tools/steering_tests/vector_testing/results/logs/%x_%j.err
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --array=0-25

# KL divergence - base_emotion and high_emotion vector types only
# Array index: layer_idx * 2 + vec_type_idx
# 13 layers × 2 vec types = 26 jobs

set -e

LAYERS=(0 5 10 15 20 25 30 35 40 45 50 55 60)
VEC_TYPES=("base_emotion_vs_others" "high_emotion_vs_others")

LAYER_IDX=$((SLURM_ARRAY_TASK_ID / 2))
VEC_IDX=$((SLURM_ARRAY_TASK_ID % 2))

LAYER=${LAYERS[$LAYER_IDX]}
VEC_TYPE=${VEC_TYPES[$VEC_IDX]}

echo "=== KL Divergence HF: Layer ${LAYER}, ${VEC_TYPE} ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
source /workspace-vast/annas/.secrets/load_secrets.sh

START_TIME=$(date +%s)

python -m steering_tests.vector_testing.kl_div_hf \
    --model google/gemma-3-27b-it \
    --layer ${LAYER} \
    --vector-dir steering_tests/vectors/gemma3_27b/${VEC_TYPE} \
    --num-samples 50 \
    --num-random 5 \
    --layer-norm-pct 1.0 \
    --scales 1 5 10 20 50 100 150

END_TIME=$(date +%s)
echo "Runtime: $((END_TIME - START_TIME)) seconds"
