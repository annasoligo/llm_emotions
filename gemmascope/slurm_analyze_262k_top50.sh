#!/bin/bash
#SBATCH --job-name=analyze_262k
#SBATCH --output=gemmascope/slurm_jobs/analyze_262k_%j.out
#SBATCH --error=gemmascope/slurm_jobs/analyze_262k_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

# Analyze R1 LoRA with 262k SAE - get top 50 A and B features

set -e

cd /workspace-vast/annas/git/research-tools

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

export HF_HOME="/workspace-vast/annas/.cache/huggingface"
export TRANSFORMERS_CACHE="/workspace-vast/annas/.cache/huggingface"

echo "Python: $(which python)"
echo "Starting 262k SAE analysis with top 50 features..."

LORA_PATH="/workspace-vast/annas/models/gemma3-27b-dpo-minimal-r1-L20/2026-01-15_18-10-31"
OUTPUT_PREFIX="gemmascope/outputs/r1_lora_analysis/gemma3-27b-dpo-minimal-r1-L20_layer20_262k_top50"

python gemmascope/analyze_r1_lora_with_sae.py \
    --lora-path "$LORA_PATH" \
    --sae-layer 20 \
    --sae-width "262k" \
    --sae-l0 "small" \
    --model-size "27b" \
    --top-k 50 \
    --output "$OUTPUT_PREFIX" \
    --load-examples \
    --use-all-layers \
    --analyze-A \
    --base-model "google/gemma-3-27b-it"

echo ""
echo "Results saved to:"
echo "  B matrix: ${OUTPUT_PREFIX}.summary.json"
echo "  A matrix: ${OUTPUT_PREFIX}.A_matrix.json"
echo "Done!"
