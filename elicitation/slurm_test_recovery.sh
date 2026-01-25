#!/bin/bash
#SBATCH --job-name=test_recovery
#SBATCH --output=slurm_jobs/test_recovery_%j.out
#SBATCH --error=slurm_jobs/test_recovery_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=dev
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -e

cd /workspace-vast/annas/git/research-tools

# Activate venv
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

# Load API keys
source /workspace-vast/annas/.secrets/load_secrets.sh

# Create output directory
mkdir -p elicitation/slurm_jobs

echo "=============================================="
echo "TESTING RECOVERY DATA GENERATION"
echo "=============================================="
echo "Time: $(date)"
echo ""

python elicitation/test_recovery_generation.py

echo ""
echo "=============================================="
echo "DONE: $(date)"
echo "=============================================="
