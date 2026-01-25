#!/bin/bash
#SBATCH --job-name=autointerp_L40
#SBATCH --output=gemmascope/slurm_jobs/autointerp_L40_%j.out
#SBATCH --error=gemmascope/slurm_jobs/autointerp_L40_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Auto-interpret top 50 A and B features from L40 262k SAE

set -e

cd /workspace-vast/annas/git/research-tools

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

export HF_HOME="/workspace-vast/annas/.cache/huggingface"
export PYTHONUNBUFFERED=1

echo "Python: $(which python)"
echo "ANTHROPIC_API_KEY set: $([ -n \"$ANTHROPIC_API_KEY\" ] && echo 'yes' || echo 'no')"

B_PATH="gemmascope/outputs/r1_lora_analysis/gemma3-27b-dpo-r1-L40-alpha128-3ep_layer40_262k.summary.json"
A_PATH="gemmascope/outputs/r1_lora_analysis/gemma3-27b-dpo-r1-L40-alpha128-3ep_layer40_262k.A_matrix.json"
OUTPUT_PATH="gemmascope/outputs/r1_lora_analysis/autointerp_L40_262k_A_and_B.json"

python gemmascope/autointerp_features.py \
    --summary-path "$B_PATH" \
    --a-matrix-path "$A_PATH" \
    --sae-layer 40 \
    --sae-width "262k" \
    --sae-l0 "small" \
    --model-size "27b" \
    --n-features 50 \
    --output "$OUTPUT_PATH" \
    --interp-model "claude-sonnet-4-20250514" \
    --concurrent 20

echo "Done! Results saved to $OUTPUT_PATH"
