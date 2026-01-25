#!/bin/bash
#SBATCH --job-name=sb_text_L45
#SBATCH --partition=highram
#SBATCH --nodelist=node-12
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/workspace-vast/annas/logs/sb_text_L45_%j.out
#SBATCH --error=/workspace-vast/annas/logs/sb_text_L45_%j.err

# Text-based steering experiment at layer 45
# Emotions: fear, happiness, anger at 100% and 150%

cd /workspace-vast/annas/git/research-tools

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

LAYER=45
NORM_PCTS="1.00 1.50"
EMOTIONS="fear happiness anger"
NUM_SAMPLES=10
MAX_PROMPTS=8
MAX_TOKENS=3000

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Detected $NUM_GPUS GPUs"

echo "Running Qwen 235B sandbagging with TEXT vectors - Layer $LAYER"
echo "Emotions: $EMOTIONS"
echo "Norm percentages: $NORM_PCTS"
echo "Max tokens: $MAX_TOKENS"

python -m experiments.steering.experiments.sandbagging_steering_qwen235b \
    --layer "$LAYER" \
    --norm-pcts $NORM_PCTS \
    --emotions $EMOTIONS \
    --num-samples "$NUM_SAMPLES" \
    --max-prompts "$MAX_PROMPTS" \
    --max-tokens "$MAX_TOKENS" \
    --vector-type text \
    --tensor-parallel "$NUM_GPUS" \
    --gpu-memory 0.90

echo "Experiment complete"
