#!/bin/bash
#SBATCH --job-name=suppress_sandbag_gemma27b_v2
#SBATCH --output=/workspace-vast/annas/logs/suppress_sandbag_gemma27b_v2_%j.out
#SBATCH --error=/workspace-vast/annas/logs/suppress_sandbag_gemma27b_v2_%j.out
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G

source /workspace-vast/annas/.secrets/load_secrets.sh

# Required for vLLM apply_model with custom callables
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "===== Sandbagging with Suppression Experiment v2 ====="
echo "Model: google/gemma-3-27b-it"
echo "Higher fear (15%) + smaller suppression multipliers"
echo "=================================================="

# Higher fear (15%) to reliably induce sandbagging
# Much smaller suppression multipliers (0.005-0.02) to avoid breaking coherence
python -m steering_tests.suppression_experiments.sandbagging_with_suppression \
    --model google/gemma-3-27b-it \
    --layers 28 29 30 31 32 \
    --fear-pct 0.15 \
    --suppress-pcts 0.005 0.01 0.02 \
    --num-samples 30 \
    --tp 1 \
    --gpu-memory 0.90 \
    --max-model-len 8192 \
    --max-tokens 2000

echo "===== Experiment Complete ====="
