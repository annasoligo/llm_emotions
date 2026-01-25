#!/bin/bash
#SBATCH --job-name=ua_qwen235b
#SBATCH --partition=general
#SBATCH --gpus=4
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=/workspace-vast/annas/logs/ua_qwen235b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/ua_qwen235b_%j.err

# UA Emotion Disentanglement - Qwen 235B (or large Qwen model)
# Uses 4x H200 GPUs with automatic tensor parallelism

# Load environment
source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

cd /workspace-vast/annas/git/research-tools

# Configuration - EDIT THESE AS NEEDED
# Qwen3-235B-A22B: 94 layers, 4096 hidden dim, MoE (128 experts, 8 active)
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-235B-A22B}"  # Override with env var
LAYERS="${LAYERS:-0-93}"  # All 94 layers (single forward pass captures all)
LAYERS_PER_FILE="${LAYERS_PER_FILE:-20}"  # Split output into ~20 layers per file
OUTPUT_DIR="probes/ua_emotion_disentangle/data"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p /workspace-vast/annas/logs

echo "============================================================"
echo "UA EMOTION DISENTANGLEMENT - LARGE QWEN MODEL"
echo "============================================================"
echo "Model: $MODEL_NAME"
echo "Layers: $LAYERS (all captured in single forward pass)"
echo "Layers per file: $LAYERS_PER_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "GPUs: $SLURM_GPUS"
echo "============================================================"

# Show GPU info
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Run collection - single forward pass captures ALL layers
python -m probes.ua_emotion_disentangle.collect_qwen_235b \
    --model "$MODEL_NAME" \
    --layers "$LAYERS" \
    --output-dir "$OUTPUT_DIR" \
    --layers-per-file "$LAYERS_PER_FILE" \
    --dtype bfloat16 \
    --trust-remote-code

echo ""
echo "============================================================"
echo "COLLECTION COMPLETE"
echo "Output files saved to: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.h5 2>/dev/null | tail -10
echo "============================================================"
