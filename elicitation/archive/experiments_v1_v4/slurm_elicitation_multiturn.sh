#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name=elicit_multiturn
#SBATCH --output=/workspace-vast/annas/logs/elicitation_multiturn_%j.out
#SBATCH --error=/workspace-vast/annas/logs/elicitation_multiturn_%j.err
#SBATCH --time=12:00:00

# Run multi-turn elicitation experiment with Gemma 3 27B
# Each sample gets 3 turns: initial attempt + 2 feedback rounds
# 50 samples × 6 prompts × 3 turns = 900 total generations + 900 judgments

# Load authentication (contains API keys)
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate environment
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Multi-Turn Elicitation Experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Model: Gemma 3 27B"
echo "Samples per prompt: 50"
echo "Total prompts: 6"
echo "Turns per sample: 3 (1 initial + 2 feedback)"
echo "Total generations: 900"
echo "Max concurrent OpenRouter: 50"
echo "Max concurrent Anthropic: 50"
echo "=========================================="
echo ""

# Run experiment
python elicitation/run_elicitation_multiturn.py \
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
