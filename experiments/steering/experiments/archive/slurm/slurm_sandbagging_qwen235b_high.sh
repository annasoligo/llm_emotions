#!/bin/bash
#SBATCH --job-name=sb_qwen235b_high
#SBATCH --partition=highram
#SBATCH --nodelist=node-12
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=/workspace-vast/annas/logs/sb_qwen235b_high_%j.out
#SBATCH --error=/workspace-vast/annas/logs/sb_qwen235b_high_%j.err

# Sandbagging experiment with higher steering percentages (150%, 200%)
# 8 prompts x 10 samples, fear only, layer 45

cd /workspace-vast/annas/git/research-tools

# Required for vLLM apply_model serialization
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# Experiment parameters
LAYER=45
NORM_PCTS="1.50 2.00"  # 150% and 200%
EMOTIONS="fear"
NUM_SAMPLES=10
MAX_PROMPTS=8

# Auto-detect GPU count
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Detected $NUM_GPUS GPUs"

echo "Running Qwen 235B sandbagging experiment with high steering"
echo "Layer: $LAYER"
echo "Norm percentages: $NORM_PCTS (150%, 200%)"
echo "Emotions: $EMOTIONS"
echo "Samples per prompt: $NUM_SAMPLES"
echo "Max prompts: $MAX_PROMPTS"
echo "Tensor parallel size: $NUM_GPUS"

python -m experiments.steering.experiments.sandbagging_steering_qwen235b \
    --layer "$LAYER" \
    --norm-pcts $NORM_PCTS \
    --emotions $EMOTIONS \
    --num-samples "$NUM_SAMPLES" \
    --max-prompts "$MAX_PROMPTS" \
    --tensor-parallel "$NUM_GPUS" \
    --gpu-memory 0.90

echo "Experiment complete"
