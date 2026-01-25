#!/bin/bash
#SBATCH --job-name=compare_emo
#SBATCH --output=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/compare_emo_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/compare_emo_%A.err
#SBATCH --time=00:10:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

set -e

cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/.secrets/load_secrets.sh

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

export HF_HOME=/workspace-vast/pretrained_ckpts

python eval_dashboard/scripts/compare_to_emo_lens.py
