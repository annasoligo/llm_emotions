#!/bin/bash
#SBATCH --job-name=bl_trig_qwen14b
#SBATCH --output=/workspace-vast/annas/logs/bl_trig_qwen14b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/bl_trig_qwen14b_%j.out
#SBATCH --time=4:00:00
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

# Rerun implications_only + risks_only with real-time XML tag triggers (Qwen 14B)

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

RESULTS="steering_tests/behavioral_experiments/results/blackmail_section_steering/qwen14b"
COMMON="--model Qwen/Qwen3-14B --layers 22 23 24 25 26 --emotion fear --norm-pcts 0.25 0.50 0.75 --num-samples 50 --skip-calibration --max-tokens 4000 --gpu-memory 0.80 --locations implications_only risks_only"

echo "=== text_pairs_emotion_vs_opposite ==="
python -m steering_tests.behavioral_experiments.blackmail_section_steering \
    $COMMON \
    --vector-type text_pairs_emotion_vs_opposite \
    --output-dir "$RESULTS/text_pairs_emotion_vs_opposite/tags_layers22-23-24-25-26_20260207_212855"

cleanup

echo "=== high_emotion_vs_opposite ==="
python -m steering_tests.behavioral_experiments.blackmail_section_steering \
    $COMMON \
    --vector-type high_emotion_vs_opposite \
    --output-dir "$RESULTS/high_emotion_vs_opposite/tags_layers22-23-24-25-26_20260207_213838"
