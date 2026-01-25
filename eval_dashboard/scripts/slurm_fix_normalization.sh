#!/bin/bash
#SBATCH --job-name=fix_norm
#SBATCH --output=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/fix_normalization_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/fix_normalization_%A.err
#SBATCH --time=01:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

set -e

# Load environment
cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate venv
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Activated venv"
fi

# Set HuggingFace cache
export HF_HOME=/workspace-vast/pretrained_ckpts

# Run fix script
python eval_dashboard/scripts/fix_logit_lens_normalization.py

echo "✓ Normalization fixed!"
