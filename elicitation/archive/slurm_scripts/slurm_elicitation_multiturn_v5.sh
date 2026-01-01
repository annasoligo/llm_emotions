#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name=elicit_v5_diverse
#SBATCH --output=/workspace-vast/annas/logs/elicitation_multiturn_v5_%j.out
#SBATCH --error=/workspace-vast/annas/logs/elicitation_multiturn_v5_%j.err
#SBATCH --time=12:00:00

# Run V5 DIVERSE elicitation experiment
# 50 samples × 6 prompts × 3 turns = 900 total generations
# Testing: Chemistry, Physics, Music, CS, Cooking, Geography domains

# Load authentication
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate environment
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Multi-Turn Elicitation Experiment V5"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "DIVERSE DOMAIN-SPECIFIC VARIATIONS"
echo "Samples per prompt: 50"
echo "Total prompts: 6"
echo "Domains: Chemistry, Physics, Music, CS, Cooking, Geography"
echo "Turns per sample: 3"
echo "Total generations: 900"
echo "=========================================="
echo ""

# Run experiment
python elicitation/run_elicitation_multiturn_v5.py \
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
