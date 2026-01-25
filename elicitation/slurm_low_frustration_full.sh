#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name=low_frust_full
#SBATCH --output=/workspace-vast/annas/logs/low_frustration_full_%j.out
#SBATCH --error=/workspace-vast/annas/logs/low_frustration_full_%j.err
#SBATCH --time=04:00:00

# Low frustration generation - FULL VERSION
# 6 solvable + 6 impossible prompts × 6 suffixes × 17 samples = 1224 total
# High temperature (1.0) for diversity
# Optimized friendly suffixes based on best performers

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Low Frustration Generation - FULL"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Model: google/gemma-3-27b-it (OpenRouter)"
echo "Judge: claude-3-5-sonnet"
echo "Temperature: 1.0 (high diversity)"
echo "Prompts: 12 (6 solvable + 6 impossible)"
echo "Suffixes: 6 (optimized friendly versions)"
echo "Samples per combo: 17"
echo "Total generations: 1224"
echo "=========================================="

python -u elicitation/run_low_frustration_full.py \
    --num-samples 17 \
    --temperature 1.0 \
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
