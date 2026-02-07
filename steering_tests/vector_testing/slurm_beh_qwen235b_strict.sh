#!/bin/bash
#SBATCH --job-name=q235b_strict
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --exclude=node-10,node-11
#SBATCH --output=results/logs/qwen235b_strict_%A_%a.out
#SBATCH --error=results/logs/qwen235b_strict_%A_%a.err
#SBATCH --array=3-6,8-9%2

# Run ALL 235B behavioral experiments with STRICT prompt
# Array jobs:
#   0: base_emotion_vs_others
#   1: base_emotion_vs_others --reversed
#   2: high_emotion_vs_others
#   3: high_emotion_vs_others --reversed
#   4: text_pairs_emotion_vs_neutral
#   5: text_pairs_emotion_vs_neutral --reversed
#   6: text_pairs_emotion_vs_opposite
#   7: text_pairs_emotion_vs_opposite --reversed
#   8: text_pairs_emotion_vs_others
#   9: text_pairs_emotion_vs_others --reversed

set -e

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# CRITICAL: Use old vLLM engine for 235B model stability
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_V1=0
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1

# Vector types
VECTOR_TYPES=(
    "base_emotion_vs_others"
    "base_emotion_vs_others"
    "high_emotion_vs_others"
    "high_emotion_vs_others"
    "text_pairs_emotion_vs_neutral"
    "text_pairs_emotion_vs_neutral"
    "text_pairs_emotion_vs_opposite"
    "text_pairs_emotion_vs_opposite"
    "text_pairs_emotion_vs_others"
    "text_pairs_emotion_vs_others"
)

REVERSED_FLAGS=(
    ""
    "--reversed"
    ""
    "--reversed"
    ""
    "--reversed"
    ""
    "--reversed"
    ""
    "--reversed"
)

VECTOR_TYPE=${VECTOR_TYPES[$SLURM_ARRAY_TASK_ID]}
REVERSED=${REVERSED_FLAGS[$SLURM_ARRAY_TASK_ID]}
VECTOR_DIR="steering_tests/vectors/qwen235b/${VECTOR_TYPE}"

echo "=== Behavioral Sweep: Qwen 235B ${VECTOR_TYPE} ${REVERSED} (STRICT PROMPT) ==="
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Vector dir: ${VECTOR_DIR}"
echo "VLLM_USE_V1: ${VLLM_USE_V1}"
echo "Node: $(hostname)"
echo "Started: $(date)"

python -m steering_tests.vector_testing.behavioral_shift \
    --model Qwen/Qwen3-235B-A22B \
    --layers 0 10 20 30 40 50 60 70 80 90 \
    --vector-dir "${VECTOR_DIR}" \
    --scales 0 1 3 5 10 15 20 \
    --num-random 3 \
    --tensor-parallel 4 \
    --coherence-threshold 0.7 \
    ${REVERSED}

echo ""
echo "============================================================"
echo "COMPLETE"
echo "============================================================"
echo "Completed: $(date)"
