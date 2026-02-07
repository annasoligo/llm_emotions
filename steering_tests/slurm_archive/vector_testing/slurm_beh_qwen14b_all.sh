#!/bin/bash
#SBATCH --job-name=qwen14b_beh
#SBATCH --output=/workspace-vast/annas/git/research-tools/steering_tests/vector_testing/results/logs/%x_%j.out
#SBATCH --error=/workspace-vast/annas/git/research-tools/steering_tests/vector_testing/results/logs/%x_%j.err
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --array=0-9

# Behavioral sweep: Qwen 14B all vector types (forward and reversed)

set -e

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
source /workspace-vast/annas/.secrets/load_secrets.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

mkdir -p steering_tests/vector_testing/results/logs

VECTOR_TYPES=("base_emotion_vs_others" "high_emotion_vs_others" "text_pairs_emotion_vs_neutral" "text_pairs_emotion_vs_opposite" "text_pairs_emotion_vs_others")
REVERSED_FLAGS=("" "--reversed")

# Calculate which vector type and reversed flag to use
VTYPE_IDX=$((SLURM_ARRAY_TASK_ID / 2))
REV_IDX=$((SLURM_ARRAY_TASK_ID % 2))

VTYPE=${VECTOR_TYPES[$VTYPE_IDX]}
REV_FLAG=${REVERSED_FLAGS[$REV_IDX]}

echo "=== Behavioral Sweep: Qwen 14B ${VTYPE} ${REV_FLAG} ==="
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Started: $(date)"

# Qwen 14B has 48 layers (0-47)
python -m steering_tests.vector_testing.behavioral_shift \
    --model Qwen/Qwen2.5-14B-Instruct \
    --layers 0 4 8 12 16 20 24 28 32 36 40 44 \
    --vector-dir steering_tests/vectors/qwen14b/${VTYPE} \
    --scales 0 1 3 5 10 15 20 \
    --num-random 3 \
    ${REV_FLAG}

echo "Completed: $(date)"
