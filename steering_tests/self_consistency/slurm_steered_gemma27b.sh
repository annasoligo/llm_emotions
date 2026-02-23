#!/bin/bash
#SBATCH --job-name=sc_steered_gemma27b
#SBATCH --output=/workspace-vast/annas/logs/sc_steered_gemma27b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/sc_steered_gemma27b_%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cleanup() {
    pkill -9 -f "vllm" || true
    pkill -9 -f "ray" || true
}
trap cleanup EXIT SIGTERM SIGINT

python3 -m steering_tests.self_consistency.generate_steered \
    --model google/gemma-3-27b-it \
    --vector-types text_pairs_emotion_vs_opposite high_emotion_vs_opposite \
    --layers 35 36 37 38 39 \
    --scales 10 20 30 \
    --samples 10 \
    --max-model-len 4096 \
    --max-tokens 512
