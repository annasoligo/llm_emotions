#!/bin/bash
#SBATCH --job-name=sandbag_textmeandiff
#SBATCH --output=/workspace-vast/annas/logs/sandbag_textmeandiff_%j.out
#SBATCH --error=/workspace-vast/annas/logs/sandbag_textmeandiff_%j.err
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

# Multi-layer anti-steering sandbagging experiment with TEXTMEANDIFF vectors
#
# Tests fear steering at layer 30 with anti-steering in layer 60
# Uses textmeandiff vectors (emotion - neutral)
#
# Note: Textmeandiff vectors only available at layers 10, 20, 25, 30, 35, 40, 45, 50, 60
# So anti-steering is only at layer 60 (unlike contrast_all which uses 57-61)
#
# Conditions:
#   1. Baseline
#   2. Fear +7.5% at layer 30 only
#   3. Fear +7.5% at layer 30, anti-steer -5% at layer 60
#   4. Fear +7.5% at layer 30, anti-steer -7.5% at layer 60

set -e

# Load secrets for HuggingFace
source /workspace-vast/annas/.secrets/load_secrets.sh

# Allow pickle serialization for steering hooks
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

echo "=== Multi-Layer Anti-Steering Sandbagging Experiment (TEXTMEANDIFF) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
date

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

mkdir -p /workspace-vast/annas/logs

python -m experiments.steering.experiments.sandbagging_multilayer_antisteer \
    --num-samples 20 \
    --main-layer 30 \
    --main-pct 0.075 \
    --antisteer-layers 60 \
    --antisteer-pcts 0.05 0.075 \
    --vector-type textmeandiff

echo "Experiment complete!"
date
