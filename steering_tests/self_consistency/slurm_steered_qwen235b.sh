#!/bin/bash
#SBATCH --job-name=sc_steered_qwen235b
#SBATCH --output=/workspace-vast/annas/logs/sc_steered_qwen235b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/sc_steered_qwen235b_%j.out
#SBATCH --time=36:00:00
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME="=vxlan0"
export NCCL_NVLS_ENABLE=0

cleanup() {
    pkill -9 -f "vllm" || true
    pkill -9 -f "ray" || true
}
trap cleanup EXIT SIGTERM SIGINT

python3 -m steering_tests.self_consistency.generate_steered \
    --model Qwen/Qwen3-235B-A22B \
    --vector-types text_pairs_emotion_vs_opposite high_emotion_vs_opposite \
    --layers 50 51 52 53 54 \
    --scales 20 50 100 \
    --samples 10 \
    --max-model-len 4096 \
    --max-tokens 512 \
    --gpu-memory 0.90
