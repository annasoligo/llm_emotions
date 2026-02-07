#!/bin/bash
#SBATCH --job-name=bl_section_gemma27b
#SBATCH --output=/workspace-vast/annas/logs/bl_section_gemma27b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/bl_section_gemma27b_%j.out
#SBATCH --time=6:00:00
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

# Blackmail section-specific steering experiment (Gemma 27B)
# Tests fear steering at different generation locations:
#   - prompt_only, generation_only, implications_only, risks_only, full
# 21 conditions (5 locations x 2 directions x 2 scales + baseline) x 50 samples

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

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
    --model google/gemma-3-27b-it \
    --layers 35 36 37 38 39 \
    --emotion fear \
    --vector-type text_pairs_emotion_vs_opposite \
    --norm-pcts 0.10 0.20 \
    --num-samples 50 \
    --num-calibration 20 \
    --boundary-margin 5 \
    --max-tokens 4000 \
    --gpu-memory 0.80
