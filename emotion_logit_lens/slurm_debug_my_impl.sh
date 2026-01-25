#!/bin/bash
#SBATCH --job-name=debug_my_impl
#SBATCH --output=/workspace-vast/annas/git/research-tools/emotion_logit_lens/debug_my_impl_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/emotion_logit_lens/debug_my_impl_%A.err
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

python emotion_logit_lens/debug_my_implementation.py
