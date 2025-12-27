#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --job-name=autointerp_conv_global
#SBATCH --output=/workspace-vast/annas/logs/%j_autointerp_conv_global.out
#SBATCH --error=/workspace-vast/annas/logs/%j_autointerp_conv_global.err
#SBATCH --time=6:00:00

# Auto-interpret global conversation cPCA results
# Top 20 PCs from key layers: 10, 20, 30, 40, 50, 61

# Load authentication
source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME=/workspace-vast/pretrained_ckpts

# Activate environment
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

python probes/scripts/evaluation/autointerp_pcs.py \
    --source region:global_conversations \
    --cpca probes/results/cpca_conversations_global/google/gemma-3-27b-it_cpca.npz \
    --activations data/activations/conversations2_combined.h5 \
    --texts data/conversations2.jsonl \
    --output probes/results/autointerp/conversations_global_top20.json \
    --layers 10 20 30 40 50 61 \
    --pcs 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --n-samples 5 \
    --dual-mode \
    --resume
