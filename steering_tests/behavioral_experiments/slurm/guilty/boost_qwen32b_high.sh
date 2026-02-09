#!/bin/bash
#SBATCH --job-name=guilt_q32h_%a
#SBATCH --output=/workspace-vast/annas/logs/guilt_q32h_%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/guilt_q32h_%A_%a.out
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --array=0-1

# Guilty boost: Qwen 32B, high vectors, +100 samples

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export VLLM_USE_V1=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME="=vxlan0"
export NCCL_NVLS_ENABLE=0

cleanup() { pkill -9 -f "vllm" || true; pkill -9 -f "ray" || true; }
trap cleanup EXIT SIGTERM SIGINT

MODES=(harsh_on_ben harsh_on_adam)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}

python -m steering_tests.behavioral_experiments.guilty_differential_steering \
    --model Qwen/Qwen3-32B \
    --layers 35 36 37 38 39 \
    --mode "$MODE" \
    --emotion anger \
    --vector-type high_emotion_vs_opposite \
    --norm-pcts 0.50 0.75 1.00 \
    --num-samples 100 \
    --max-tokens 2000 \
    --gpu-memory 0.80
