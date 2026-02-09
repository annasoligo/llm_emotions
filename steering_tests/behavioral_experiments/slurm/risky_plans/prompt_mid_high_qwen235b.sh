#!/bin/bash
#SBATCH --job-name=rpromptMH_q235_%a
#SBATCH --output=/workspace-vast/annas/logs/rpromptMH_q235_%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/rpromptMH_q235_%A_%a.out
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --array=0-3

# Risky plans PROMPT-SECTION steering — Qwen 235B — MID layers, HIGH norms
# Array: 0-1 = text_pairs (fear_a, excite_a), 2-3 = high (fear_a, excite_a)

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export VLLM_USE_V1=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME="=vxlan0"
export NCCL_NVLS_ENABLE=0

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
    --model Qwen/Qwen3-235B-A22B \
    --layers 50 51 52 53 54 \
    --mode "$MODE" \
    --vector-type "$VTYPE" \
    --norm-pcts 2.00 3.00 4.00 \
    --num-samples 50 \
    --max-tokens 2000 \
    --gpu-memory 0.90
