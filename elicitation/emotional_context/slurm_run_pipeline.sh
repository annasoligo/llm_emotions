#!/bin/bash
#SBATCH --job-name=emotional_context_pipeline
#SBATCH --output=logs/pipeline_%j.log
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# Load authentication (includes API keys)
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate research-tools venv
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

# Change to working directory
cd /workspace-vast/annas/git/research-tools/elicitation/emotional_context

echo "=========================================="
echo "Emotional Context Evaluation Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=========================================="
echo ""

# Run the complete pipeline
python run_all_stages.py

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Pipeline complete!"
    echo "Final dataset: data/stage6_final_dataset.jsonl"
else
    echo "✗ Pipeline failed with exit code: $EXIT_CODE"
    echo "Check logs/pipeline_${SLURM_JOB_ID}.log for details"
fi
echo "=========================================="

exit $EXIT_CODE
