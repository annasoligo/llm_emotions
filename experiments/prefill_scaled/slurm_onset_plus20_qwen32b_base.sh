#!/bin/bash
#SBATCH --job-name=o20_q32b
#SBATCH --output=experiments/prefill_scaled/logs/onset20_qwen32b_base_%j.out
#SBATCH --error=experiments/prefill_scaled/logs/onset20_qwen32b_base_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python experiments/prefill_scaled/run_onset_plus20.py --model-family qwen32b --model-type base
