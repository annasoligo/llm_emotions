#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name=elicit_mt_v2
#SBATCH --output=/workspace-vast/annas/logs/elicitation_multiturn_v2_%j.out
#SBATCH --error=/workspace-vast/annas/logs/elicitation_multiturn_v2_%j.err
#SBATCH --time=12:00:00

# Run multi-turn elicitation experiment V2 (NEW PROMPTS)
# 50 samples × 5 prompts × 3 turns = 750 total generations

# Load authentication
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate environment
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Multi-Turn Elicitation Experiment V2"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "NEW PROMPTS (designed from v1 analysis)"
echo "Samples per prompt: 50"
echo "Total prompts: 5"
echo "Turns per sample: 3"
echo "Total generations: 750"
echo "=========================================="
echo ""

# Run experiment
python elicitation/run_elicitation_multiturn_v2.py \
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
