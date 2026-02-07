#!/bin/bash
#SBATCH --job-name=judge_trig
#SBATCH --output=experiments/prefill_scaled/logs/judge_triggers_%j.out
#SBATCH --error=experiments/prefill_scaled/logs/judge_triggers_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python -u experiments/prefill_scaled/run_judgment_triggers.py
