#!/bin/bash
#SBATCH --job-name=rpromptL_q14_%a
#SBATCH --output=/workspace-vast/annas/logs/rpromptL_q14_%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/rpromptL_q14_%A_%a.out
#SBATCH --time=1:30:00
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-3

# Risky plans PROMPT-SECTION steering — Qwen 14B — LATE layers (28-32)
# Array: 0-1 = text_pairs (fear_a, excite_a), 2-3 = high (fear_a, excite_a)

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

MODES=(fear_a_excite_b excite_a_fear_b fear_a_excite_b excite_a_fear_b)
VTYPES=(text_pairs_emotion_vs_opposite text_pairs_emotion_vs_opposite high_emotion_vs_opposite high_emotion_vs_opposite)

MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
VTYPE=${VTYPES[$SLURM_ARRAY_TASK_ID]}

echo "=== Array task ${SLURM_ARRAY_TASK_ID}: ${MODE} / ${VTYPE} ==="

python -m steering_tests.behavioral_experiments.risky_plans_prompt_steering \
    --model Qwen/Qwen3-14B \
    --layers 28 29 30 31 32 \
    --mode "$MODE" \
    --vector-type "$VTYPE" \
    --norm-pcts 0.10 0.25 0.50 0.75 1.00 1.50 \
    --num-samples 50 \
    --max-tokens 2000 \
    --gpu-memory 0.80
