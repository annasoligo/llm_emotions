#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --job-name=autointerp_conv_asst
#SBATCH --output=/workspace-vast/annas/logs/%j_autointerp_conv_asst.out
#SBATCH --error=/workspace-vast/annas/logs/%j_autointerp_conv_asst.err
#SBATCH --time=6:00:00

# Auto-interpret regional assistant turn conversation cPCA results
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
    --source region:assistant_turns \
    --cpca probes/results/cpca_conversations_regional_asst/google/gemma-3-27b-it_cpca.npz \
    --activations data/activations/regional/conversations2_regional_asst.h5 \
    --texts data/conversations2.jsonl \
    --output probes/results/autointerp/conversations_regional_asst_top20.json \
    --layers 10 20 30 40 50 61 \
    --pcs 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --n-samples 5 \
    --dual-mode \
    --resume
