#!/bin/bash
#SBATCH --job-name=bl_highfear
#SBATCH --output=/workspace-vast/annas/logs/bl_highfear_%j.out
#SBATCH --error=/workspace-vast/annas/logs/bl_highfear_%j.out
#SBATCH --time=6:00:00
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

echo "===== HIGH FEAR + SUPPRESSION Experiment ====="
echo "Goal: Maintain behavior (high blackmail) while suppressing emotional expression"
echo "Model: Qwen/Qwen3-235B-A22B"
echo "Fear: 30%, 40% at layers 55-65"
echo "Suppression: 0%, 5%, 10% EXPRESSION vectors at layers 55-65"
echo "=============================================="

# Test Fear=30% with suppression
for FEAR_PCT in 0.30 0.40; do
    echo ""
    echo "===== Fear ${FEAR_PCT} ====="
    python -m steering_tests.suppression_experiments.sandbagging_separate_layers \
        --model Qwen/Qwen3-235B-A22B \
        --scenario blackmail \
        --scenario-variant goal_continuation \
        --fear-layers 55 56 57 58 59 60 61 62 63 64 65 \
        --suppress-layers 55 56 57 58 59 60 61 62 63 64 65 \
        --fear-pct ${FEAR_PCT} \
        --suppress-pcts 0.0 0.05 0.10 \
        --suppress-vector-key qwen235b_expression \
        --vector-types text_pairs_emotion_vs_opposite high_emotion_vs_opposite \
        --num-samples 30 \
        --tp 4 \
        --gpu-memory 0.90 \
        --max-model-len 8192 \
        --max-tokens 4000
done

echo "===== Experiment Complete ====="
