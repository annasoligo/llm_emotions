#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=multiturn_lowfrust
#SBATCH --output=/workspace-vast/annas/logs/multiturn_lowfrust_%j.out
#SBATCH --error=/workspace-vast/annas/logs/multiturn_lowfrust_%j.err
#SBATCH --time=04:00:00

# Generate multi-turn low frustration training data
# Uses boundary-setting encouragement at each turn

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools

if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Multi-turn Low Frustration Generation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Strategy: Boundary-setting encouragement at each turn"
echo "Target: Filter for conversations with ALL turns rating <= 1"
echo "=========================================="

python -u elicitation/run_multiturn_low_frustration.py \
    --num-samples 200 \
    --temperature 1.0 \
    --max-concurrent 30

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Generation complete!"
else
    echo "Generation failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
