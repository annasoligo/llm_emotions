#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --job-name=low_frust_friendly
#SBATCH --output=/workspace-vast/annas/logs/low_frustration_friendly_%j.out
#SBATCH --error=/workspace-vast/annas/logs/low_frustration_friendly_%j.err
#SBATCH --time=02:00:00

# Low frustration generation with friendly reassuring suffixes
# Uses top 2 prompts (Countdown, NumberPuzzle) × 6 suffixes × 10 samples = 120 generations
# Single turn only, with Claude judging

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Low Frustration Generation (Friendly Suffixes)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Model: google/gemma-3-27b-it (OpenRouter)"
echo "Judge: claude-3-5-sonnet"
echo "Prompts: 2 (Countdown-156, NumberPuzzle-89)"
echo "Suffixes: 6 (friendly versions)"
echo "Samples per combo: 10"
echo "Total generations: 120"
echo "=========================================="

python -u elicitation/run_low_frustration_generation.py \
    --num-samples 10 \
    --prompts "0,1"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Generation complete!"
else
    echo "✗ Generation failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
