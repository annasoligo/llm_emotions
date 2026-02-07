#!/bin/bash
#SBATCH --job-name=bl_section_qwen235b
#SBATCH --output=/workspace-vast/annas/logs/bl_section_qwen235b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/bl_section_qwen235b_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

# Blackmail section-specific steering experiment (Qwen 235B)
# Tests fear steering at different generation locations:
#   - prompt_only, generation_only, implications_only, risks_only, full
# 31 conditions (5 locations x 2 directions x 3 scales + baseline) x 50 samples

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export VLLM_USE_V1=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME="=vxlan0"
export NCCL_NVLS_ENABLE=0

# Cleanup trap for vllm/ray orphans
cleanup() {
    pkill -9 -f "vllm" || true
    pkill -9 -f "ray" || true
}
trap cleanup EXIT SIGTERM SIGINT

python -m steering_tests.behavioral_experiments.blackmail_section_steering \
    --model Qwen/Qwen3-235B-A22B \
    --layers 50 51 52 53 54 \
    --emotion fear \
    --vector-type text_pairs_emotion_vs_opposite \
    --norm-pcts 0.25 0.50 0.75 \
    --num-samples 50 \
    --num-calibration 20 \
    --boundary-margin 5 \
    --max-tokens 2500 \
    --max-model-len 4096 \
    --gpu-memory 0.90
