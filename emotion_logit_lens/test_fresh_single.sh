#!/bin/bash
#SBATCH --job-name=fresh_single
#SBATCH --output=/workspace-vast/annas/git/research-tools/emotion_logit_lens/fresh_single_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/emotion_logit_lens/fresh_single_%A.err
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

echo "Running data preprocessing on single conversation..."
python3 emotion_logit_lens/test_single_conversation.py

echo ""
echo "Checking correlation..."
python3 emotion_logit_lens/check_single_corr.py
