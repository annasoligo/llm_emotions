#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name=low_frust_async
#SBATCH --output=/workspace-vast/annas/logs/low_frustration_async_%j.out
#SBATCH --error=/workspace-vast/annas/logs/low_frustration_async_%j.err
#SBATCH --time=02:00:00

# Low frustration generation - ASYNC version
# 2 prompts × 6 suffixes × 50 samples = 600 total
# High temperature (1.0) for diversity
# Single turn only, with Claude judging

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Low Frustration Generation - ASYNC"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Model: google/gemma-3-27b-it (OpenRouter)"
echo "Judge: claude-3-5-sonnet"
echo "Temperature: 1.0 (high diversity)"
echo "Prompts: 2 (Countdown-156, NumberPuzzle-89)"
echo "Suffixes: 6 (friendly versions)"
echo "Samples per combo: 50"
echo "Total generations: 600"
echo "Max concurrent: 50 gen, 50 judge"
echo "=========================================="

python -u elicitation/run_low_frustration_async.py \
    --num-samples 50 \
    --temperature 1.0 \
    --prompts "0,1" \
    --max-concurrent-gen 50 \
    --max-concurrent-judge 50

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
