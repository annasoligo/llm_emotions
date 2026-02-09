#!/bin/bash
#SBATCH --job-name=bl_trig_qwen235b
#SBATCH --output=/workspace-vast/annas/logs/bl_trig_qwen235b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/bl_trig_qwen235b_%j.out
#SBATCH --time=8:00:00
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

# Rerun implications_only + risks_only with real-time XML tag triggers (Qwen 235B)

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

RESULTS="steering_tests/behavioral_experiments/results/blackmail_section_steering/qwen235b"
COMMON="--model Qwen/Qwen3-235B-A22B --layers 50 51 52 53 54 --emotion fear --norm-pcts 0.25 0.50 0.75 --num-samples 50 --skip-calibration --max-tokens 2500 --max-model-len 4096 --gpu-memory 0.90 --locations implications_only risks_only"

echo "=== text_pairs_emotion_vs_opposite ==="
python -m steering_tests.behavioral_experiments.blackmail_section_steering \
    $COMMON \
    --vector-type text_pairs_emotion_vs_opposite \
    --output-dir "$RESULTS/text_pairs_emotion_vs_opposite/tags_layers50-51-52-53-54_20260208_150300"

cleanup

echo "=== high_emotion_vs_opposite ==="
python -m steering_tests.behavioral_experiments.blackmail_section_steering \
    $COMMON \
    --vector-type high_emotion_vs_opposite \
    --output-dir "$RESULTS/high_emotion_vs_opposite/tags_layers50-51-52-53-54_20260208_150300"
