#!/bin/bash
#SBATCH --job-name=ratio_pca
#SBATCH --output=probes/logs/ratio_pca_%j.log
#SBATCH --error=probes/logs/ratio_pca_%j.log
#SBATCH --partition=luna-compute
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -e

echo "========================================"
echo "RATIO PCA - Tier Data"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo ""

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Run ratio PCA with k=10 (remove top-10 neutral-dominated PCs)
python -m probes.scripts.run_ratio_pca \
    --activations data/activations/texts_combined.h5 \
    --output probes/results/ratio_pca_tier_data/ \
    --k 10 \
    --n-components 50 \
    --n-pcs-all 100

echo ""
echo "Completed at: $(date)"
