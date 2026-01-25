#!/bin/bash
#SBATCH --job-name=recompute_logit
#SBATCH --output=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/recompute_logit_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/recompute_logit_%A.err
#SBATCH --time=01:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

set -e

cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/.secrets/load_secrets.sh

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Activated venv"
fi

export HF_HOME=/workspace-vast/pretrained_ckpts

echo "================================"
echo "RECOMPUTING LOGIT LENS WITH FIXED LAYER RANGE (40-50)"
echo "================================"

python eval_dashboard/scripts/add_logit_lens_to_existing.py

echo "================================"
echo "✓ Done! Logit lens recomputed with layers 40-50"
echo "================================"
