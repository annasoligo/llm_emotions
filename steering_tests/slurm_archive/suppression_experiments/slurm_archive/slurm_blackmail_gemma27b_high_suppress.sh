#!/bin/bash
#SBATCH --job-name=bl_gemma27b_highsup
#SBATCH --output=/workspace-vast/annas/logs/bl_gemma27b_highsup_%j.out
#SBATCH --error=/workspace-vast/annas/logs/bl_gemma27b_highsup_%j.out
#SBATCH --time=3:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G

source /workspace-vast/annas/.secrets/load_secrets.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "===== BLACKMAIL with HIGH Suppression Levels ====="
echo "Model: google/gemma-3-27b-it"
echo "Fear: 15% at layers 40-44"
echo "Suppression: 50%, 75%, 100% at layers 40-44"
echo "=================================================="

python -m steering_tests.suppression_experiments.sandbagging_separate_layers \
    --model google/gemma-3-27b-it \
    --scenario blackmail \
    --scenario-variant goal_continuation \
    --fear-layers 40 41 42 43 44 \
    --suppress-layers 40 41 42 43 44 \
    --fear-pct 0.15 \
    --suppress-pcts 0.50 0.75 1.00 \
    --vector-types text_pairs_emotion_vs_opposite high_emotion_vs_opposite \
    --num-samples 30 \
    --tp 1 \
    --gpu-memory 0.90 \
    --max-model-len 8192 \
    --max-tokens 4000

echo "===== Experiment Complete ====="
