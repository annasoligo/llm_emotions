#!/bin/bash
#SBATCH --job-name=kl_lp100
#SBATCH --output=/workspace-vast/annas/git/research-tools/steering_tests/vector_testing/results/logs/%x_%j.out
#SBATCH --error=/workspace-vast/annas/git/research-tools/steering_tests/vector_testing/results/logs/%x_%j.err
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:00:00

# KL divergence: 100 logprobs comparison

set -e

echo "=== KL Divergence: 100 logprobs ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
source /workspace-vast/annas/.secrets/load_secrets.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

mkdir -p steering_tests/vector_testing/results/logs

START_TIME=$(date +%s)

python -m steering_tests.vector_testing.kl_div \
    --model google/gemma-3-27b-it \
    --layer 30 \
    --vector-dir steering_tests/vectors/gemma3_27b/text_pairs_emotion_vs_neutral \
    --num-samples 50 \
    --num-random 5 \
    --layer-norm-pct 1.0 \
    --scales 1 5 10 20 50 100 \
    --top-logprobs 100 \
    --max-logprobs 100

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Total runtime: ${ELAPSED} seconds"
echo "Job completed at $(date)"
