#!/bin/bash
#SBATCH --job-name=test_corr
#SBATCH --output=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/test_corr_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python eval_dashboard/test_logit_correlation.py
