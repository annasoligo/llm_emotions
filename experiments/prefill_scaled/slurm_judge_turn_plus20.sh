#!/bin/bash
#SBATCH --job-name=judge_t20
#SBATCH --output=experiments/prefill_scaled/logs/judge_turn_plus20_%j.out
#SBATCH --error=experiments/prefill_scaled/logs/judge_turn_plus20_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python -u experiments/prefill_scaled/run_judgment_turn_plus20.py
