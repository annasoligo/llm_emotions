#!/bin/bash
#SBATCH --job-name=short_prompts
#SBATCH --output=/workspace-vast/annas/logs/short_prompts_%j.out
#SBATCH --error=/workspace-vast/annas/logs/short_prompts_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

# Load secrets (API keys)
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate virtual environment
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

# Navigate to repo root
cd /workspace-vast/annas/git/research-tools

# Run the experiment
echo "Starting short prompts experiment at $(date)"
echo "Model: google/gemma-3-27b-it"
echo "Design: 27 scenarios × 19 emotions (11 discrete + 8 dimensional) × 3 paraphrases = 1,539 prompts"
echo "Samples per prompt: 10"
echo "Total API calls: 15,390"
echo "Mode: TAG-FIRST (max 20 tokens per response)"
echo ""

python3 experiments/behavior_tests/run_short_prompts_experiment.py

echo ""
echo "Completed at $(date)"
