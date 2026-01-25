#!/bin/bash
#SBATCH --job-name=sandbag_tmd_full
#SBATCH --output=/workspace-vast/annas/logs/sandbag_tmd_full_%j.out
#SBATCH --error=/workspace-vast/annas/logs/sandbag_tmd_full_%j.err
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

# Multi-layer anti-steering sandbagging experiment with TEXTMEANDIFF vectors
# Using layers 57-61 for anti-steering (same as contrast_all)
#
# Conditions:
#   1. Baseline
#   2. Fear +7.5% at layer 30 only
#   3. Fear +7.5% at layer 30, anti-steer -5% at layers 57-61
#   4. Fear +7.5% at layer 30, anti-steer -7.5% at layers 57-61

set -e

# Load secrets for HuggingFace
source /workspace-vast/annas/.secrets/load_secrets.sh

# Allow pickle serialization for steering hooks
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

echo "=== Multi-Layer Anti-Steering Sandbagging Experiment (TEXTMEANDIFF FULL) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
date

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Verify vectors exist
echo "Checking for required vectors..."
for LAYER in 57 58 59 60 61; do
    VEC_FILE="experiments/steering/vectors/all_emotions_textmeandiff_layer${LAYER}.npz"
    if [ ! -f "$VEC_FILE" ]; then
        echo "ERROR: Missing vector file: $VEC_FILE"
        exit 1
    fi
    echo "  Found: $VEC_FILE"
done

mkdir -p /workspace-vast/annas/logs

python -m experiments.steering.experiments.sandbagging_multilayer_antisteer \
    --num-samples 20 \
    --main-layer 30 \
    --main-pct 0.075 \
    --antisteer-layers 57 58 59 60 61 \
    --antisteer-pcts 0.05 0.075 \
    --vector-type textmeandiff

echo "Experiment complete!"
date
