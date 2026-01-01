#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name=elicit_gemma27b
#SBATCH --output=/workspace-vast/annas/logs/elicitation_gemma27b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/elicitation_gemma27b_%j.err
#SBATCH --time=8:00:00

# Run elicitation experiment with Gemma 3 27B
# 50 samples per prompt × 5 prompts = 250 total samples
# With 50 concurrent requests for both OpenRouter and Anthropic

# Load authentication (contains API keys)
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate environment
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Elicitation Experiment - Gemma 3 27B"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Samples per prompt: 50"
echo "Total prompts: 5"
echo "Total samples: 250"
echo "Max concurrent OpenRouter: 50"
echo "Max concurrent Anthropic: 50"
echo "=========================================="
echo ""

# Run experiment
python elicitation/run_elicitation_concurrent.py \
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
