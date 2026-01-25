#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=calm_variants
#SBATCH --output=/workspace-vast/annas/logs/calm_variants_%j.out
#SBATCH --error=/workspace-vast/annas/logs/calm_variants_%j.err
#SBATCH --time=02:00:00

# Generate calm responses for variant prompts (to enable more DPO pairs)

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools

if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Diverse Calm Generation for Variant Prompts"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Target: 200 conversations per variant prompt (1000 total)"
echo "=========================================="

python elicitation/run_diverse_calm_variants.py \
    --model "google/gemma-3-27b-it" \
    --judge-model "claude-3-5-sonnet-20241022" \
    --samples-per-prompt 200 \
    --max-concurrent 50 \
    --temperature 1.0

echo ""
echo "=========================================="
echo "Generation complete!"
echo "=========================================="
