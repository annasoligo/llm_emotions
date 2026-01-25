#!/bin/bash
#SBATCH --job-name=test_logit_20
#SBATCH --output=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/test_logit_20_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/test_logit_20_%A.err
#SBATCH --time=01:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1

set -e

cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/.secrets/load_secrets.sh

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Activated venv"
fi

export HF_HOME=/workspace-vast/pretrained_ckpts

python3 eval_dashboard/recompute_logit_incremental_simple.py
