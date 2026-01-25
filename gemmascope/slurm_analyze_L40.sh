#!/bin/bash
#SBATCH --job-name=analyze_L40
#SBATCH --output=gemmascope/slurm_jobs/analyze_L40_%j.out
#SBATCH --error=gemmascope/slurm_jobs/analyze_L40_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

# Analyze R1 L40 LoRA with L40 SAE - get top 50 A and B features

set -e

cd /workspace-vast/annas/git/research-tools

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

export HF_HOME="/workspace-vast/annas/.cache/huggingface"
export TRANSFORMERS_CACHE="/workspace-vast/annas/.cache/huggingface"
export PYTHONUNBUFFERED=1

echo "Python: $(which python)"
echo "Starting L40 SAE analysis with top 50 features..."

LORA_PATH="/workspace-vast/annas/models/gemma3-27b-dpo-r1-L40-alpha128-3ep/2026-01-16_10-41-00"
OUTPUT_PREFIX="gemmascope/outputs/r1_lora_analysis/gemma3-27b-dpo-r1-L40-alpha128-3ep_layer40_262k"

# Analyze with 262k SAE at layer 40 (standard layer available for Gemma Scope 2)
python gemmascope/analyze_r1_lora_with_sae.py \
    --lora-path "$LORA_PATH" \
    --sae-layer 40 \
    --sae-width "262k" \
    --sae-l0 "small" \
    --model-size "27b" \
    --top-k 50 \
    --output "$OUTPUT_PREFIX" \
    --load-examples \
    --analyze-A \
    --base-model "google/gemma-3-27b-it"

echo ""
echo "Results saved to:"
echo "  B matrix: ${OUTPUT_PREFIX}.summary.json"
echo "  A matrix: ${OUTPUT_PREFIX}.A_matrix.json"
echo "Done analysis!"
