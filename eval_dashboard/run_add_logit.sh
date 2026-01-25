#!/bin/bash
#SBATCH --job-name=add_logit_norm
#SBATCH --output=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/add_logit_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
python eval_dashboard/add_logit_lens_to_existing.py
