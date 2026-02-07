#!/bin/bash
#SBATCH --job-name=trig_dpo
#SBATCH --output=experiments/prefill_scaled/logs/triggers_dpo_%j.out
#SBATCH --error=experiments/prefill_scaled/logs/triggers_dpo_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python -u experiments/prefill_scaled/run_triggers_generation.py --model-family dpo_calm --model-type dpo
