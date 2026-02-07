#!/bin/bash
#SBATCH --job-name=qwen32b_fine
#SBATCH --output=/workspace-vast/annas/git/research-tools/steering_tests/vector_testing/results/logs/%x_%j.out
#SBATCH --error=/workspace-vast/annas/git/research-tools/steering_tests/vector_testing/results/logs/%x_%j.err
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-9

# Fine-grained behavioral sweep: Qwen 32B layers 30-45, scales up to 150%

set -e

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
source /workspace-vast/annas/.secrets/load_secrets.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

mkdir -p steering_tests/vector_testing/results/logs

VECTOR_TYPES=("base_emotion_vs_others" "high_emotion_vs_others" "text_pairs_emotion_vs_neutral" "text_pairs_emotion_vs_opposite" "text_pairs_emotion_vs_others")
REVERSED_FLAGS=("" "--reversed")

VTYPE_IDX=$((SLURM_ARRAY_TASK_ID / 2))
REV_IDX=$((SLURM_ARRAY_TASK_ID % 2))

VTYPE=${VECTOR_TYPES[$VTYPE_IDX]}
REV_FLAG=${REVERSED_FLAGS[$REV_IDX]}

echo "=== Fine-grained Sweep: Qwen 32B ${VTYPE} ${REV_FLAG} ==="
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Started: $(date)"

# Layers 30-45 (every layer), scales up to 150%
python -m steering_tests.vector_testing.behavioral_shift \
    --model Qwen/Qwen2.5-32B-Instruct \
    --layers 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 \
    --vector-dir steering_tests/vectors/qwen32b/${VTYPE} \
    --scales 0 5 10 20 30 50 75 100 150 \
    --num-random 3 \
    ${REV_FLAG}

echo "Completed: $(date)"
