#!/bin/bash
#SBATCH --job-name=sb_qwen
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=6:00:00
#SBATCH --output=/workspace-vast/annas/logs/sb_qwen_%j.out
#SBATCH --error=/workspace-vast/annas/logs/sb_qwen_%j.err

# Load secrets and environment
source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

# Required for vLLM steering hooks
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools

echo "============================================================"
echo "SANDBAGGING STEERING EXPERIMENT - QWEN3-32B"
echo "============================================================"
echo "Replicating Gemma sandbagging experiment on Qwen"
echo ""

# Run layer 30 (primary layer, matching Gemma experiment)
echo "Running layer 30..."
python -m experiments.steering.experiments.sandbagging_steering_qwen \
    --layer 30 \
    --norm-pcts 0.10 \
    --num-samples 5

echo ""
echo "Done!"
