#!/bin/bash
#SBATCH --job-name=analyze_r1_lora
#SBATCH --output=gemmascope/slurm_jobs/analyze_r1_lora_%j.out
#SBATCH --error=gemmascope/slurm_jobs/analyze_r1_lora_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Analyze R1 LoRA with Gemma Scope SAE

set -e

cd /workspace-vast/annas/git/research-tools

# Load secrets (HF_TOKEN, etc.)
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate venv with ML packages
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

# Set writable HF cache directory
export HF_HOME="/workspace-vast/annas/.cache/huggingface"
export TRANSFORMERS_CACHE="/workspace-vast/annas/.cache/huggingface"
mkdir -p "$HF_HOME"

echo "Python: $(which python)"
echo "HF_TOKEN set: $([ -n \"$HF_TOKEN\" ] && echo 'yes' || echo 'no')"
echo "HF_HOME: $HF_HOME"

# Create output directory
OUTPUT_DIR="gemmascope/outputs/r1_lora_analysis"
mkdir -p "$OUTPUT_DIR"
mkdir -p gemmascope/slurm_jobs

# Available R1 LoRA models
LORA_MODELS=(
    "/workspace-vast/annas/models/gemma3-27b-dpo-minimal-r1-L20/2026-01-15_18-10-31"
    "/workspace-vast/annas/models/gemma3-27b-dpo-minimal-r1-L20-2ep/2026-01-15_18-45-53"
)

# SAE configurations to try
# For 27B-IT:
#   - Standard layers (resid_post): 16, 31, 40, 53
#   - All layers (resid_post_all): 0-61, but only 16k/262k widths, small/big L0

# For exact layer match with LoRA (layer 20) - use _all variant
SAE_LAYER_EXACT=20
SAE_WIDTHS_ALL=("16k" "262k")  # Only these available for _all
SAE_L0_ALL="small"  # Only small/big available, no medium

# For nearby standard layer (layer 16 is closest to 20)
SAE_LAYERS_STD=(16 31)  # Standard layers near layer 20
SAE_WIDTHS_STD=("16k" "65k" "262k")
SAE_L0_STD="medium"

USE_ALL_LAYERS="--use-all-layers"  # Flag for exact layer analysis

echo "Starting R1 LoRA analysis with Gemma Scope SAEs"
echo "================================================"

# Analyze each LoRA
for LORA_PATH in "${LORA_MODELS[@]}"; do
    LORA_NAME=$(basename "$(dirname "$LORA_PATH")")_$(basename "$LORA_PATH")
    echo ""
    echo "Analyzing: $LORA_NAME"
    echo "Path: $LORA_PATH"

    # 1. Exact layer match (layer 20) using resid_post_all
    echo ""
    echo "=== Exact layer analysis (layer ${SAE_LAYER_EXACT}, using _all variant) ==="
    for SAE_WIDTH in "${SAE_WIDTHS_ALL[@]}"; do
        OUTPUT_PREFIX="${OUTPUT_DIR}/${LORA_NAME}_layer${SAE_LAYER_EXACT}_${SAE_WIDTH}_all"

        echo "  SAE: layer=${SAE_LAYER_EXACT}, width=${SAE_WIDTH}, l0=${SAE_L0_ALL} (all-layers variant)"

        python gemmascope/analyze_r1_lora_with_sae.py \
            --lora-path "$LORA_PATH" \
            --sae-layer "$SAE_LAYER_EXACT" \
            --sae-width "$SAE_WIDTH" \
            --sae-l0 "$SAE_L0_ALL" \
            --model-size "27b" \
            --top-k 30 \
            --output "$OUTPUT_PREFIX" \
            --load-examples \
            --show-bottom-tokens \
            $USE_ALL_LAYERS \
            2>&1 | tee "${OUTPUT_PREFIX}.log"

        echo "  -> Saved to ${OUTPUT_PREFIX}.*"
    done

    # 2. Standard layer analysis (nearby standard layers: 16, 31)
    echo ""
    echo "=== Standard layer analysis (nearby layers) ==="
    for SAE_LAYER in "${SAE_LAYERS_STD[@]}"; do
        for SAE_WIDTH in "${SAE_WIDTHS_STD[@]}"; do
            OUTPUT_PREFIX="${OUTPUT_DIR}/${LORA_NAME}_layer${SAE_LAYER}_${SAE_WIDTH}_std"

            echo "  SAE: layer=${SAE_LAYER}, width=${SAE_WIDTH}, l0=${SAE_L0_STD} (standard variant)"

            python gemmascope/analyze_r1_lora_with_sae.py \
                --lora-path "$LORA_PATH" \
                --sae-layer "$SAE_LAYER" \
                --sae-width "$SAE_WIDTH" \
                --sae-l0 "$SAE_L0_STD" \
                --model-size "27b" \
                --top-k 30 \
                --output "$OUTPUT_PREFIX" \
                --load-examples \
                --show-bottom-tokens \
                2>&1 | tee "${OUTPUT_PREFIX}.log"

            echo "  -> Saved to ${OUTPUT_PREFIX}.*"
        done
    done
done

echo ""
echo "Analysis complete!"
echo "Results in: $OUTPUT_DIR"
