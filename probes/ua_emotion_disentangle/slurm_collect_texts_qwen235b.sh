#!/bin/bash
#SBATCH --job-name=text_act_qwen235b
#SBATCH --partition=highram
#SBATCH --nodelist=node-12
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=/workspace-vast/annas/logs/text_act_qwen235b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/text_act_qwen235b_%j.err

# Collect activations from emotional text pairs for Qwen 235B
# Uses HuggingFace with device_map='auto' across 4 GPUs
# Collects ALL 94 layers in single forward pass, batched for efficiency

cd /workspace-vast/annas/git/research-tools

# Output path
OUTPUT_DIR="probes/ua_emotion_disentangle/activations"
mkdir -p $OUTPUT_DIR

OUTPUT_FILE="${OUTPUT_DIR}/qwen235b_text_activations.h5"
BATCH_SIZE=4  # Conservative for memory with 94 layers

echo "Collecting text activations for Qwen 235B"
echo "All 94 layers in single pass"
echo "Batch size: $BATCH_SIZE"
echo "Output: $OUTPUT_FILE"

python -m probes.ua_emotion_disentangle.collect_texts_qwen235b \
    --input outputs/data/texts_combined_pairs.jsonl \
    --output "$OUTPUT_FILE" \
    --position last \
    --batch-size $BATCH_SIZE

echo "Collection complete"
