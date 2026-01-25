#!/bin/bash
#SBATCH --job-name=diverse_pcs_20_40
#SBATCH --output=/workspace-vast/annas/logs/%j.out
#SBATCH --error=/workspace-vast/annas/logs/%j.err
#SBATCH --partition=general
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Compute opposite-source emotion PCs for layers 20-40

set -e

echo "========================================"
echo "COMPUTING OPPOSITE-SOURCE EMOTION PCS (LAYERS 20-40)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo ""

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Generate layer string for all layers 20-40
LAYERS=$(seq -s ' ' 20 40)

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
