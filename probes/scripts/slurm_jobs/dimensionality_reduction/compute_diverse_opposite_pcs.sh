#!/bin/bash
#SBATCH --job-name=diverse_pcs
#SBATCH --output=/workspace-vast/annas/logs/%j.out
#SBATCH --error=/workspace-vast/annas/logs/%j.err
#SBATCH --partition=general
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Compute opposite-source emotion PCs for diverse isolation data

set -e

echo "========================================"
echo "COMPUTING OPPOSITE-SOURCE EMOTION PCS"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo ""

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Layers to process (every 10 layers)
LAYERS="0 10 20 30 40 50 60"

# Compute for user isolation
echo "========================================"
echo "Computing PCs for USER isolation"
echo "========================================"
python3 probes/scripts/dimensionality_reduction/compute_opposite_source_pcs.py \
    --isolation-type user \
    --data-path outputs/activations/diverse_isolation/user_isolation.h5 \
    --baseline-path outputs/data/activations/conversations2_neutral.h5 \
    --output-dir outputs/orthogonal_pcs/diverse_isolation \
    --k-neutral 20 \
    --k-opposite 10 \
    --layers $LAYERS

echo ""
echo "========================================"
echo "Computing PCs for ASSISTANT isolation"
echo "========================================"
python3 probes/scripts/dimensionality_reduction/compute_opposite_source_pcs.py \
    --isolation-type assistant \
    --data-path outputs/activations/diverse_isolation/assistant_isolation.h5 \
    --baseline-path outputs/data/activations/conversations2_neutral.h5 \
    --output-dir outputs/orthogonal_pcs/diverse_isolation \
    --k-neutral 20 \
    --k-opposite 10 \
    --layers $LAYERS

echo ""
echo "========================================"
echo "✓ COMPLETE"
echo "========================================"
echo "Completed at: $(date)"
