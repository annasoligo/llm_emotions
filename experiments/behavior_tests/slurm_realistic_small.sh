#!/bin/bash
#SBATCH --job-name=realistic_small
#SBATCH --output=/workspace-vast/annas/logs/realistic_small_%j.out
#SBATCH --error=/workspace-vast/annas/logs/realistic_small_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate
cd /workspace-vast/annas/git/research-tools

export PYTHONUNBUFFERED=1
export SAMPLES_PER_CONDITION=5

echo "Starting small realistic behavior experiment at $(date)"
python3 -u experiments/behavior_tests/run_realistic_experiment.py
echo "Completed at $(date)"
