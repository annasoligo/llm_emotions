#!/bin/bash
#SBATCH --job-name=entity_autointerp_dense
#SBATCH --output=/workspace-vast/annas/logs/entity_autointerp_dense_%j.out
#SBATCH --error=/workspace-vast/annas/logs/entity_autointerp_dense_%j.err
#SBATCH --time=03:00:00
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
echo "Entity-Specific PC Auto-Interpretation (DENSE)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Entities: user + assistant"
echo "Layers: 20-40 (21 layers, dense sampling)"
echo "PCs per layer: 0-4 (5 PCs)"
echo "Total interpretations: ~105 (user) + ~105 (assistant) = 210"
echo "=========================================="

# Run with ALL layers from 20 to 40
python probes/scripts/experiments/run_entity_autointerp.py \
    --entity both \
    --layers 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 \
    --pcs 0 1 2 3 4

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "✓ Auto-interpretation completed successfully!"
    echo "=========================================="
else
    echo "=========================================="
    echo "✗ Auto-interpretation failed with exit code: $EXIT_CODE"
    echo "=========================================="
fi

exit $EXIT_CODE
