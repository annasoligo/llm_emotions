#!/bin/bash
#SBATCH --job-name=orthogonal_pcs_all
#SBATCH --partition=general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/workspace-vast/annas/logs/%j.out
#SBATCH --error=/workspace-vast/annas/logs/%j.err

# Compute neutral and shared orthogonal PCs for all 62 layers
# Runtime: ~15 sec/layer × 62 layers = ~15 minutes

set -e

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "=========================================="

# Activate environment
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo ""
echo "Computing orthogonal PCs for controlled variation (all 62 layers)..."
echo ""

python3 -m probes.scripts.dimensionality_reduction.compute_controlled_variation_orthogonal_pcs \
    --user-data outputs/activations/controlled_variation/user_isolation.h5 \
    --asst-data outputs/activations/controlled_variation/assistant_isolation.h5 \
    --layer-range 0 62 \
    --k-neutral 20 \
    --k-shared 10 \
    --output outputs/orthogonal_pcs/controlled_variation

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
