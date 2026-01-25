#!/bin/bash
#SBATCH --job-name=autointerp
#SBATCH --output=gemmascope/slurm_jobs/autointerp_%j.out
#SBATCH --error=gemmascope/slurm_jobs/autointerp_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=general
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Auto-interpret SAE features aligned with R1 LoRA

set -e

cd /workspace-vast/annas/git/research-tools

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

export HF_HOME="/workspace-vast/annas/.cache/huggingface"

echo "Python: $(which python)"
echo "ANTHROPIC_API_KEY set: $([ -n \"$ANTHROPIC_API_KEY\" ] && echo 'yes' || echo 'no')"

SUMMARY_PATH="gemmascope/outputs/r1_lora_analysis/gemma3-27b-dpo-minimal-r1-L20_2026-01-15_18-10-31_layer20_16k_all.summary.json"
OUTPUT_PATH="gemmascope/outputs/r1_lora_analysis/autointerp_layer20_16k.json"

python gemmascope/autointerp_features.py \
    --summary-path "$SUMMARY_PATH" \
    --sae-layer 20 \
    --sae-width "16k" \
    --sae-l0 "small" \
    --use-all-layers \
    --model-size "27b" \
    --n-features 15 \
    --output "$OUTPUT_PATH" \
    --interp-model "claude-sonnet-4-20250514"

echo "Done!"
