#!/bin/bash
#SBATCH --job-name=compute_baselines
#SBATCH --output=/workspace-vast/annas/logs/compute_baselines_%j.out
#SBATCH --error=/workspace-vast/annas/logs/compute_baselines_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

# Compute per-layer probe baselines from WildChat data
# No GPU needed - just CPU for matrix operations

set -e

echo "=========================================="
echo "COMPUTE PER-LAYER PROBE BASELINES"
echo "=========================================="
echo "Start time: $(date)"
echo "Host: $(hostname)"

# Activate venv
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

# Use shared HuggingFace cache
export HF_HOME=/workspace-vast/pretrained_ckpts

# Show Python info
echo "Python: $(which python)"
python --version

cd /workspace-vast/annas/git/research-tools

# Create output directory for slurm logs
mkdir -p eval_dashboard/slurm_jobs

# Run baseline computation
python eval_dashboard/compute_probe_baselines.py

echo ""
echo "=========================================="
echo "DONE!"
echo "End time: $(date)"
echo "=========================================="
