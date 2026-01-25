#!/bin/bash
#SBATCH --job-name=entity_autointerp_early_sparse
#SBATCH --output=/workspace-vast/annas/logs/entity_autointerp_early_sparse_%j.out
#SBATCH --error=/workspace-vast/annas/logs/entity_autointerp_early_sparse_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

# Load secrets for Claude API
source /workspace-vast/annas/.secrets/load_secrets.sh

# Change to project directory
cd /workspace-vast/annas/git/research-tools

# Activate virtual environment if it exists
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Entity-Specific PC Auto-Interpretation (EARLY SPARSE)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Entities: user + assistant"
echo "Layers: 5, 10, 15 (3 early layers)"
echo "PCs per layer: 0-4 (5 PCs)"
echo "Total interpretations: ~15 (user) + ~15 (assistant) = 30"
echo "=========================================="

# Run with early sparse layers
python probes/scripts/experiments/run_entity_autointerp.py \
    --entity both \
    --layers 5 10 15 \
    --pcs 0 1 2 3 4 \
    --output-dir outputs/interpretations/autointerp/controlled_variation/early

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "✓ Early sparse auto-interpretation completed successfully!"
    echo "=========================================="
else
    echo "=========================================="
    echo "✗ Auto-interpretation failed with exit code: $EXIT_CODE"
    echo "=========================================="
fi

exit $EXIT_CODE
