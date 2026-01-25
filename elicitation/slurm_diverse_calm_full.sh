#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=diverse_calm
#SBATCH --output=/workspace-vast/annas/logs/diverse_calm_%j.out
#SBATCH --error=/workspace-vast/annas/logs/diverse_calm_%j.err
#SBATCH --time=02:00:00

# Generate diverse calm multi-turn data with varied system prompts

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools

if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Diverse Calm Data Generation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Strategy: 8 different calm/positive system prompts"
echo "Target: 1000 conversations (200 per puzzle)"
echo "=========================================="

python -u elicitation/run_diverse_calm_generation.py \
    --num-samples 200 \
    --num-turns 3 \
    --max-concurrent 50 \
    --temperature 1.0

echo ""
echo "=========================================="
echo "Generation complete!"
echo "=========================================="
