#!/bin/bash
#SBATCH --job-name=acts_multi_model
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/multi_model_%j.out
#SBATCH --error=logs/multi_model_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Example: Collect activations on multiple models (large model with multi-GPU)

# Configuration
MODEL="Qwen/Qwen3-32B"
INPUT="steering_tests/data/emotion_prompts_MODEL_500.jsonl"
OUTPUT="steering_tests/activations/emotion_prompts_qwen3_32b"
LAYERS="all"  # Collect all layers

echo "====================================================="
echo "Multi-GPU Activation Collection"
echo "====================================================="
echo "Model: $MODEL"
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo "Layers: $LAYERS"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "====================================================="

# Create output directory
mkdir -p steering_tests/activations
mkdir -p logs

# Run collection (PyTorch hooks work great with multi-GPU!)
python steering_tests/activation_collection/collect.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --mode chat \
  --layers "$LAYERS" \
  --dtype bfloat16 \
  --batch-save 100 \
  --resume

echo "====================================================="
echo "DONE!"
echo "====================================================="
