#!/bin/bash
#SBATCH --job-name=analyze_A
#SBATCH --output=gemmascope/slurm_jobs/analyze_A_%j.out
#SBATCH --error=gemmascope/slurm_jobs/analyze_A_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

# Analyze A matrix of R1 LoRA via W_down @ A^T

set -e

cd /workspace-vast/annas/git/research-tools

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

export HF_HOME="/workspace-vast/annas/.cache/huggingface"
export TRANSFORMERS_CACHE="/workspace-vast/annas/.cache/huggingface"

echo "Python: $(which python)"
echo "Starting A matrix analysis..."

LORA_PATH="/workspace-vast/annas/models/gemma3-27b-dpo-minimal-r1-L20/2026-01-15_18-10-31"
OUTPUT_DIR="gemmascope/outputs/r1_lora_analysis"

python gemmascope/analyze_r1_lora_with_sae.py \
    --lora-path "$LORA_PATH" \
    --sae-layer 20 \
    --sae-width "16k" \
    --sae-l0 "small" \
    --model-size "27b" \
    --top-k 20 \
    --load-examples \
    --show-bottom-tokens \
    --use-all-layers \
    --analyze-A \
    --base-model "google/gemma-3-27b-it"

echo "Done!"
