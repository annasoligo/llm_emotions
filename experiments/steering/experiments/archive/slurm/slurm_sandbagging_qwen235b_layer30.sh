#!/bin/bash
#SBATCH --job-name=sb_qwen_L30
#SBATCH --partition=highram
#SBATCH --nodelist=node-12
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/workspace-vast/annas/logs/sb_qwen235b_L30_%j.out
#SBATCH --error=/workspace-vast/annas/logs/sb_qwen235b_L30_%j.err

# Multi-emotion experiment at layer 30
# 8 prompts x 10 samples, anger/fear/joy at 100%

cd /workspace-vast/annas/git/research-tools

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

LAYER=30
NORM_PCTS="1.00"
EMOTIONS="anger fear joy"
NUM_SAMPLES=10
MAX_PROMPTS=8
MAX_TOKENS=3000

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Detected $NUM_GPUS GPUs"

echo "Running Qwen 235B sandbagging - Layer $LAYER"
echo "Emotions: $EMOTIONS"
echo "Max tokens: $MAX_TOKENS"

python -m experiments.steering.experiments.sandbagging_steering_qwen235b \
    --layer "$LAYER" \
    --norm-pcts $NORM_PCTS \
    --emotions $EMOTIONS \
    --num-samples "$NUM_SAMPLES" \
    --max-prompts "$MAX_PROMPTS" \
    --max-tokens "$MAX_TOKENS" \
    --tensor-parallel "$NUM_GPUS" \
    --gpu-memory 0.90

echo "Experiment complete"
