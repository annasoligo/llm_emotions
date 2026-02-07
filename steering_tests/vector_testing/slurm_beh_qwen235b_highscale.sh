#!/bin/bash
#SBATCH --job-name=qwen235b_highscale
#SBATCH --output=/workspace-vast/annas/git/research-tools/steering_tests/vector_testing/results/logs/%x_%j_%a.out
#SBATCH --error=/workspace-vast/annas/git/research-tools/steering_tests/vector_testing/results/logs/%x_%j_%a.err
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=6:00:00
#SBATCH --array=0-9
#SBATCH --exclude=node-10,node-11

# Behavioral sweep: Qwen 235B HIGH SCALES (30, 50, 100, 150)
# Both forward and reversed directions

set -e

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
source /workspace-vast/annas/.secrets/load_secrets.sh

# CRITICAL: Use old vLLM engine for 235B model stability
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_V1=0
export NCCL_P2P_DISABLE=1

mkdir -p steering_tests/vector_testing/results/logs

VECTOR_TYPES=("base_emotion_vs_others" "high_emotion_vs_others" "text_pairs_emotion_vs_neutral" "text_pairs_emotion_vs_opposite" "text_pairs_emotion_vs_others")
REVERSED_FLAGS=("" "--reversed")

# Calculate which vector type and reversed flag to use
VTYPE_IDX=$((SLURM_ARRAY_TASK_ID / 2))
REV_IDX=$((SLURM_ARRAY_TASK_ID % 2))

VTYPE=${VECTOR_TYPES[$VTYPE_IDX]}
REV_FLAG=${REVERSED_FLAGS[$REV_IDX]}

echo "=== Behavioral Sweep: Qwen 235B HIGH SCALES ${VTYPE} ${REV_FLAG} ==="
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"

# Qwen 235B - use same layers as before
python -m steering_tests.vector_testing.behavioral_shift \
    --model Qwen/Qwen3-235B-A22B \
    --layers 0 10 20 30 40 50 60 70 80 90 \
    --vector-dir steering_tests/vectors/qwen235b/${VTYPE} \
    --scales 30 50 100 150 \
    --num-random 3 \
    --tensor-parallel 4 \
    --coherence-threshold 0.5 \
    ${REV_FLAG}

echo "Completed: $(date)"
