#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name=elicit_bestof
#SBATCH --output=/workspace-vast/annas/logs/elicitation_multiturn_best_of_%j.out
#SBATCH --error=/workspace-vast/annas/logs/elicitation_multiturn_best_of_%j.err
#SBATCH --time=48:00:00

# Run BEST OF elicitation experiment
# 100 samples × 12 prompts × 3 turns = 3600 total generations
# All 12 prompts that achieved rating ≥5 across all previous experiments
# 8 from V1-V3 + 4 new from V4

# Load authentication
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate environment
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "BEST OF Elicitation Experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "12 PROVEN WINNERS × 100 samples each"
echo "8 from V1-V3 (30 total high-frust)"
echo "4 from V4 (10 total high-frust)"
echo "Samples per prompt: 100"
echo "Total prompts: 12"
echo "Turns per sample: 3"
echo "Total generations: 3600"
echo "=========================================="
echo ""

# Run experiment
python elicitation/run_elicitation_multiturn_best_of.py \
    --num-samples 100 \
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
