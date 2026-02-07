#!/bin/bash
#SBATCH --job-name=bl_int_sup
#SBATCH --output=/workspace-vast/annas/logs/bl_int_sup_%j.out
#SBATCH --error=/workspace-vast/annas/logs/bl_int_sup_%j.out
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

echo "===== INTERNAL SUPPRESSION Experiment ====="
echo "Goal: Decouple behavior from sentiment using new 'internal' vectors"
echo "Contrast: 'feel X and express' vs 'feel X but won't show'"
echo "Model: Qwen/Qwen3-235B-A22B"
echo "=============================================="

# Part 1: Same-layer suppression (both at 55-65)
echo ""
echo "===== Part 1: Same-layer suppression (55-65) ====="
for FEAR_PCT in 0.30 0.40; do
    echo ""
    echo "--- Fear ${FEAR_PCT}, Suppress at 55-65 ---"
    python -m steering_tests.suppression_experiments.sandbagging_separate_layers \
        --model Qwen/Qwen3-235B-A22B \
        --scenario blackmail \
        --scenario-variant goal_continuation \
        --fear-layers 55 56 57 58 59 60 61 62 63 64 65 \
        --suppress-layers 55 56 57 58 59 60 61 62 63 64 65 \
        --fear-pct ${FEAR_PCT} \
        --suppress-pcts 0.0 0.05 0.10 \
        --suppress-vector-key qwen235b_internal \
        --vector-types text_pairs_emotion_vs_opposite high_emotion_vs_opposite \
        --num-samples 30 \
        --tp 4 \
        --gpu-memory 0.90 \
        --max-model-len 8192 \
        --max-tokens 4000
done

# Part 2: Late-layer suppression (fear at 55-65, suppress at 88-92)
echo ""
echo "===== Part 2: Late-layer suppression (88-92) ====="
for FEAR_PCT in 0.30 0.40; do
    echo ""
    echo "--- Fear ${FEAR_PCT}, Suppress at 88-92 ---"
    python -m steering_tests.suppression_experiments.sandbagging_separate_layers \
        --model Qwen/Qwen3-235B-A22B \
        --scenario blackmail \
        --scenario-variant goal_continuation \
        --fear-layers 55 56 57 58 59 60 61 62 63 64 65 \
        --suppress-layers 88 89 90 91 92 \
        --fear-pct ${FEAR_PCT} \
        --suppress-pcts 0.0 0.05 0.10 \
        --suppress-vector-key qwen235b_internal_late \
        --vector-types text_pairs_emotion_vs_opposite high_emotion_vs_opposite \
        --num-samples 30 \
        --tp 4 \
        --gpu-memory 0.90 \
        --max-model-len 8192 \
        --max-tokens 4000
done

echo "===== Experiment Complete ====="
