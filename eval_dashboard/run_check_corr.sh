#!/bin/bash
#SBATCH --job-name=check_corr
#SBATCH --output=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/check_corr_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
python eval_dashboard/check_token_vs_sentence_corr.py
