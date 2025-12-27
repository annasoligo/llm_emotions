#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --job-name=autointerp
#SBATCH --output=/workspace-vast/annas/logs/%j.out
#SBATCH --error=/workspace-vast/annas/logs/%j.err

# Generic autointerp job - CPU only (calls Claude API)
#
# Usage:
#   sbatch run_autointerp.sh --cpca <path> --activations <path> --texts <path> --output <path> --layers <layers> --pcs <pcs>
#
# Example:
#   sbatch run_autointerp.sh \
#     --cpca probes/results/cpca_tier_data_high_alpha/google/gemma-3-27b-it_cpca.npz \
#     --activations data/activations/texts_combined.h5 \
#     --texts data/texts_combined_pairs.jsonl \
#     --output probes/results/autointerp/gemma_layer30.json \
#     --layers 30 \
#     --pcs 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19

# Load authentication
source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME=/workspace-vast/pretrained_ckpts

# Activate environment
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Use region:all to use all samples (no tier filtering)
python probes/scripts/evaluation/autointerp_pcs.py \
    --source region:all \
    "$@"
