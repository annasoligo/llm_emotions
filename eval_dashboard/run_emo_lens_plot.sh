#!/bin/bash
#SBATCH --job-name=emo_lens_plot
#SBATCH --output=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/emo_lens_plot_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
python eval_dashboard/run_emo_lens_plot.py
