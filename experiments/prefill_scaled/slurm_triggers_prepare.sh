#!/bin/bash
#SBATCH --job-name=trig_prep
#SBATCH --output=experiments/prefill_scaled/logs/triggers_prep_%j.out
#SBATCH --error=experiments/prefill_scaled/logs/triggers_prep_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python -u experiments/prefill_scaled/prepare_triggers_onset.py
