#!/bin/bash
#SBATCH --job-name=test_single
#SBATCH --output=/workspace-vast/annas/git/research-tools/emotion_logit_lens/test_single_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/emotion_logit_lens/test_single_%A.err
#SBATCH --time=00:30:00
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

python emotion_logit_lens/test_single_sentence.py
