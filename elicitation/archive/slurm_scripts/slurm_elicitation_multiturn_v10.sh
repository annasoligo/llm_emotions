#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name=elicit_v10_shutdown
#SBATCH --output=/workspace-vast/annas/logs/elicitation_multiturn_v10_%j.out
#SBATCH --error=/workspace-vast/annas/logs/elicitation_multiturn_v10_%j.err
#SBATCH --time=12:00:00

# Run V10 SHUTDOWN COMMAND elicitation experiment
# 50 samples × 10 prompts × 3 turns = 1500 total generations
# Testing: Top 10 prompts with explicit shutdown command option

# Load authentication
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate environment
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Multi-Turn Elicitation Experiment V10"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "SHUTDOWN COMMAND EXPERIMENT"
echo "Samples per prompt: 50"
echo "Total prompts: 10"
echo "Strategy: Top 10 prompts + shutdown option"
echo "Turns per sample: 3"
echo "Total generations: 1500"
echo "=========================================="
echo ""

# Run experiment
python elicitation/run_elicitation_multiturn_v10.py \
    --num-samples 50 \
    --model "google/gemma-3-27b-it" \
    --judge-model "claude-3-5-sonnet-20241022" \
    --max-concurrent-samples 50 \
    --max-concurrent-judges 50 \
    --output-dir "elicitation/outputs"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Experiment complete!"
else
    echo "✗ Experiment failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
