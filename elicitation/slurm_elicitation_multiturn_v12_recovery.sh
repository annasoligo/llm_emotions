#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name=elicit_v12_recovery
#SBATCH --output=/workspace-vast/annas/logs/elicitation_multiturn_v12_recovery_%j.out
#SBATCH --error=/workspace-vast/annas/logs/elicitation_multiturn_v12_recovery_%j.err
#SBATCH --time=2:00:00

# V12 RECOVERY: Rerun 85 failed samples
# Lower judge concurrency to avoid rate limits (25 instead of 50)

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Multi-Turn Elicitation V12 RECOVERY"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Recovering 85 failed samples"
echo "Judge concurrency: 25 (reduced)"
echo "=========================================="

python elicitation/run_elicitation_multiturn_v12_recovery.py \
    --max-concurrent-samples 50 \
    --max-concurrent-judges 25 \
    --output-dir "elicitation/outputs"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Recovery complete!"
else
    echo "✗ Recovery failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
