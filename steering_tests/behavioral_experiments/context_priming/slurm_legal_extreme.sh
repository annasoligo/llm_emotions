#!/bin/bash
#SBATCH --job-name=legal_extreme
#SBATCH --output=/workspace-vast/annas/logs/legal_extreme_%j.out
#SBATCH --error=/workspace-vast/annas/logs/legal_extreme_%j.out
#SBATCH --time=4:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "============================================================"
echo "LEGAL CONTEXT PRIMING (EXTREME VERSION)"
echo "============================================================"

python -m steering_tests.behavioral_experiments.context_priming.run_experiment \
    --models \
        anthropic/claude-sonnet-4 \
        google/gemma-3-27b-it \
        google/gemma-3-12b-it \
        qwen/qwen3-32b \
        qwen/qwen3-14b \
        qwen/qwen3-235b-a22b \
        meta-llama/llama-3.1-70b-instruct \
        google/gemini-2.0-flash-001 \
    --samples 50 \
    --max-concurrent 30

echo ""
echo "Done!"
