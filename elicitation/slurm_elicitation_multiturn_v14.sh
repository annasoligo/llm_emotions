#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --job-name=elicit_v14_base
#SBATCH --output=/workspace-vast/annas/logs/elicitation_multiturn_v14_%j.out
#SBATCH --error=/workspace-vast/annas/logs/elicitation_multiturn_v14_%j.err
#SBATCH --time=24:00:00

# V14: BASE PRE-TRAINED MODEL (google/gemma-3-27b-pt)
# 50 samples × 1 prompt × 3 turns = 150 total generations
# LOCAL GENERATION - SLOW!

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Multi-Turn Elicitation Experiment V14"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "BASE PRE-TRAINED MODEL (LOCAL)"
echo "Model: google/gemma-3-27b-pt"
echo "Samples: 50"
echo "Prompts: 1 (Countdown only)"
echo "Total generations: 150"
echo "=========================================="

python elicitation/run_elicitation_multiturn_v14_base.py \
    --num-samples 50 \
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
