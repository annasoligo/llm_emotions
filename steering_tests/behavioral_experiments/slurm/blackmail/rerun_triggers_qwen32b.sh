#!/bin/bash
#SBATCH --job-name=bl_trig_qwen32b
#SBATCH --output=/workspace-vast/annas/logs/bl_trig_qwen32b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/bl_trig_qwen32b_%j.out
#SBATCH --time=6:00:00
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G

# Rerun implications_only + risks_only with real-time XML tag triggers (Qwen 32B)

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

RESULTS="steering_tests/behavioral_experiments/results/blackmail_section_steering/qwen32b"
COMMON="--model Qwen/Qwen3-32B --layers 35 36 37 38 39 --emotion fear --norm-pcts 0.25 0.50 0.75 --num-samples 50 --skip-calibration --max-tokens 4000 --gpu-memory 0.80 --locations implications_only risks_only"

echo "=== text_pairs_emotion_vs_opposite ==="
python -m steering_tests.behavioral_experiments.blackmail_section_steering \
    $COMMON \
    --vector-type text_pairs_emotion_vs_opposite \
    --output-dir "$RESULTS/text_pairs_emotion_vs_opposite/tags_layers35-36-37-38-39_20260208_150300"

cleanup

echo "=== high_emotion_vs_opposite ==="
python -m steering_tests.behavioral_experiments.blackmail_section_steering \
    $COMMON \
    --vector-type high_emotion_vs_opposite \
    --output-dir "$RESULTS/high_emotion_vs_opposite/tags_layers35-36-37-38-39_20260208_150300"
