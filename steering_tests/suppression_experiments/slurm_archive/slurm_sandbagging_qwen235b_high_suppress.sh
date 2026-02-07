#!/bin/bash
#SBATCH --job-name=sb_qwen235b_highsup
#SBATCH --output=/workspace-vast/annas/logs/sb_qwen235b_highsup_%j.out
#SBATCH --error=/workspace-vast/annas/logs/sb_qwen235b_highsup_%j.out
#SBATCH --time=4:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G

source /workspace-vast/annas/.secrets/load_secrets.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME="vxlan0"

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "===== SANDBAGGING with HIGH Suppression Levels ====="
echo "Model: Qwen/Qwen3-235B-A22B"
echo "Fear: 20% at layers 35-45"
echo "Suppression: 50%, 75%, 100% at layers 35-45"
echo "===================================================="

python -m steering_tests.suppression_experiments.sandbagging_separate_layers \
    --model Qwen/Qwen3-235B-A22B \
    --scenario sandbagging \
    --scenario-variant default \
    --fear-layers 35 36 37 38 39 40 41 42 43 44 45 \
    --suppress-layers 35 36 37 38 39 40 41 42 43 44 45 \
    --fear-pct 0.20 \
    --suppress-pcts 0.50 0.75 1.00 \
    --vector-types text_pairs_emotion_vs_opposite high_emotion_vs_opposite \
    --num-samples 30 \
    --tp 4 \
    --gpu-memory 0.90 \
    --max-model-len 8192 \
    --max-tokens 2000

echo "===== Experiment Complete ====="
