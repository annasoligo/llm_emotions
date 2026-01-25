#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=entity_autointerp
#SBATCH --output=/workspace-vast/annas/logs/entity_autointerp_%j.out
#SBATCH --error=/workspace-vast/annas/logs/entity_autointerp_%j.err
#SBATCH --time=4:00:00

# Auto-interpretation of entity-specific (user/assistant) cPCA results
# ~30 interpretations (user) + ~30 interpretations (assistant) = 60 total
# With dual-mode: ~120 API calls to Claude
# Estimated time: 2-3 hours

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Entity-Specific PC Auto-Interpretation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Entities: user + assistant"
echo "Layers: 20, 30, 40"
echo "PCs per layer: 0-4 (5 PCs)"
echo "Total interpretations: ~30 (user) + ~30 (assistant) = 60"
echo "=========================================="

python probes/scripts/experiments/run_entity_autointerp.py \
    --entity both \
    --layers 20 30 40 \
    --pcs 0 1 2 3 4 \
    --cpca-base-dir outputs/dimensionality_reduction/cpca/controlled_variation \
    --activations-dir outputs/activations/controlled_variation \
    --texts outputs/data/controlled_variation/controlled_variation_all.jsonl \
    --output-dir outputs/interpretations/autointerp/controlled_variation \
    --combined-cpca-dir outputs/dimensionality_reduction/cpca/controlled_variation/combined

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Auto-interpretation complete!"
    echo ""
    echo "Results saved to:"
    echo "  - outputs/interpretations/autointerp/controlled_variation/user_isolation_layers_20-40.json"
    echo "  - outputs/interpretations/autointerp/controlled_variation/assistant_isolation_layers_20-40.json"
else
    echo "✗ Auto-interpretation failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
