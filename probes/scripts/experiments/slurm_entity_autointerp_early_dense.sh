#!/bin/bash
#SBATCH --job-name=entity_autointerp_early_dense
#SBATCH --output=/workspace-vast/annas/logs/entity_autointerp_early_dense_%j.out
#SBATCH --error=/workspace-vast/annas/logs/entity_autointerp_early_dense_%j.err
#SBATCH --time=02:30:00
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
echo "Entity-Specific PC Auto-Interpretation (EARLY DENSE)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Entities: user + assistant"
echo "Layers: 0-19 (20 layers, dense sampling)"
echo "PCs per layer: 0-4 (5 PCs)"
echo "Total interpretations: ~100 (user) + ~100 (assistant) = 200"
echo "=========================================="

# Run with all early layers 0-19
python probes/scripts/experiments/run_entity_autointerp.py \
    --entity both \
    --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --pcs 0 1 2 3 4 \
    --output-dir outputs/interpretations/autointerp/controlled_variation/early

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "✓ Early dense auto-interpretation completed successfully!"
    echo "=========================================="
else
    echo "=========================================="
    echo "✗ Auto-interpretation failed with exit code: $EXIT_CODE"
    echo "=========================================="
fi

exit $EXIT_CODE
