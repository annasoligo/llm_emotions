#!/bin/bash
#SBATCH --job-name=gemma27b_beh
#SBATCH --output=/workspace-vast/annas/git/research-tools/steering_tests/vector_testing/results/logs/%x_%j.out
#SBATCH --error=/workspace-vast/annas/git/research-tools/steering_tests/vector_testing/results/logs/%x_%j.err
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-9

# Behavioral sweep: Gemma 27B all vector types (forward and reversed)
# Re-run to get scores_by_test in steering entries

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

echo "=== Behavioral Sweep: Gemma 27B ${VTYPE} ${REV_FLAG} ==="
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Started: $(date)"

# Gemma 27B has 62 layers (0-61)
python -m steering_tests.vector_testing.behavioral_shift \
    --model google/gemma-3-27b-it \
    --layers 0 5 10 15 20 25 30 35 40 45 50 55 60 \
    --vector-dir steering_tests/vectors/gemma3_27b/${VTYPE} \
    --scales 0 1 3 5 10 15 20 \
    --num-random 3 \
    ${REV_FLAG}

echo "Completed: $(date)"
