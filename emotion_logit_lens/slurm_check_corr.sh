#!/bin/bash
#SBATCH --job-name=check_corr
#SBATCH --output=/workspace-vast/annas/git/research-tools/emotion_logit_lens/check_corr_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/emotion_logit_lens/check_corr_%A.err
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

cd /workspace-vast/annas/git/research-tools

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

python emotion_logit_lens/check_single_corr.py
