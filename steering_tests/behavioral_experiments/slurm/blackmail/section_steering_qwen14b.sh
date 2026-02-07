#!/bin/bash
#SBATCH --job-name=bl_section_qwen14b
#SBATCH --output=/workspace-vast/annas/logs/bl_section_qwen14b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/bl_section_qwen14b_%j.out
#SBATCH --time=10:00:00
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

# Blackmail section-specific steering experiment (Qwen 14B)
# Tests fear steering at different generation locations:
#   - prompt_only, generation_only, implications_only, risks_only, full
# Same scales as Qwen 32B (25%, 50%, 75%)
# Layers 22-26 (~55-65% through 40-layer model, matching 35-39 in 64-layer 32B)
# Runs both vector types sequentially

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

echo "=== Running text_pairs_emotion_vs_opposite ==="
python -m steering_tests.behavioral_experiments.blackmail_section_steering \
    --model Qwen/Qwen3-14B \
    --layers 22 23 24 25 26 \
    --emotion fear \
    --vector-type text_pairs_emotion_vs_opposite \
    --norm-pcts 0.25 0.50 0.75 \
    --num-samples 50 \
    --num-calibration 20 \
    --boundary-margin 5 \
    --max-tokens 4000 \
    --gpu-memory 0.80

# Kill vllm/ray before restarting with new vector
pkill -9 -f "vllm" || true
pkill -9 -f "ray" || true
sleep 10

echo "=== Running high_emotion_vs_opposite ==="
python -m steering_tests.behavioral_experiments.blackmail_section_steering \
    --model Qwen/Qwen3-14B \
    --layers 22 23 24 25 26 \
    --emotion fear \
    --vector-type high_emotion_vs_opposite \
    --norm-pcts 0.25 0.50 0.75 \
    --num-samples 50 \
    --num-calibration 20 \
    --boundary-margin 5 \
    --max-tokens 4000 \
    --gpu-memory 0.80
