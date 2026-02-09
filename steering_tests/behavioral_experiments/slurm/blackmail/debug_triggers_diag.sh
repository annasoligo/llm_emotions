#!/bin/bash
#SBATCH --job-name=diag_trig
#SBATCH --output=/workspace-vast/annas/logs/diag_trig_%j.out
#SBATCH --error=/workspace-vast/annas/logs/diag_trig_%j.out
#SBATCH --time=0:30:00
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G

# Tiny diagnostic: 2 samples at 3200% with diagnostic prints in hooks.
# Check the log for [DIAG embed_hook] and [DIAG steer_hook] lines.

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

python -m steering_tests.behavioral_experiments.blackmail_section_steering \
    --model Qwen/Qwen3-32B \
    --layers 35 36 37 38 39 \
    --emotion fear \
    --norm-pcts 32.0 \
    --num-samples 2 \
    --skip-calibration \
    --max-tokens 500 \
    --gpu-memory 0.80 \
    --locations implications_only \
    --vector-type high_emotion_vs_opposite \
    --output-dir steering_tests/behavioral_experiments/results/blackmail_section_steering/qwen32b/debug_diag
