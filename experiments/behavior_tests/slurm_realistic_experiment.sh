#!/bin/bash
#SBATCH --job-name=realistic_exp
#SBATCH --output=/workspace-vast/annas/logs/realistic_experiment_%j.out
#SBATCH --error=/workspace-vast/annas/logs/realistic_experiment_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

# Load secrets (API keys)
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate virtual environment
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

# Navigate to repo root
cd /workspace-vast/annas/git/research-tools

# Configuration
export SAMPLES_PER_CONDITION="${SAMPLES_PER_CONDITION:-10}"

# Run the experiment
echo "Starting realistic behavior experiment at $(date)"
echo "Target model: google/gemma-3-27b-it"
echo "Judge model: anthropic/claude-sonnet-4"
echo "Samples per condition: $SAMPLES_PER_CONDITION"
echo ""

python3 experiments/behavior_tests/run_realistic_experiment.py

echo ""
echo "Completed at $(date)"
