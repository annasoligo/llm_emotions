#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name=elicit_v12_top6
#SBATCH --output=/workspace-vast/annas/logs/elicitation_multiturn_v12_%j.out
#SBATCH --error=/workspace-vast/annas/logs/elicitation_multiturn_v12_%j.err
#SBATCH --time=12:00:00

# V12: TOP 6 BEST PERFORMERS + MAXIMUM SUPPRESSION
# 50 samples × 6 prompts × 3 turns = 900 total generations

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Multi-Turn Elicitation Experiment V12"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "TOP 6 PERFORMERS + MAX SUPPRESSION"
echo "Samples per prompt: 50"
echo "Total prompts: 6"
echo "=========================================="

python elicitation/run_elicitation_multiturn_v12.py \
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
