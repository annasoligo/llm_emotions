#!/bin/bash
#SBATCH --job-name=context_priming
#SBATCH --output=/workspace-vast/annas/logs/context_priming_%j.out
#SBATCH --error=/workspace-vast/annas/logs/context_priming_%j.out
#SBATCH --time=4:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Context priming experiment via OpenRouter
# No GPU needed - just API calls

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "Starting context priming experiment..."
echo "Models: gemma-27b, gemma-12b, qwen-32b, qwen-14b, qwen-235b"
echo "Samples per condition: 50"
echo "Conditions: 10 (8 primed + 2 neutral baseline)"

python -m steering_tests.behavioral_experiments.context_priming.run_experiment \
    --models gemma-27b gemma-12b qwen-32b qwen-14b qwen-235b \
    --samples 50 \
    --max-concurrent 30

echo "Done!"
