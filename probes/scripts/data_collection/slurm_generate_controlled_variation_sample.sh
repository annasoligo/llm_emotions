#!/bin/bash
#SBATCH --job-name=gen_controlled_var
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/gen_controlled_var_%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/gen_controlled_var_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Change to project directory
cd /workspace-vast/annas/git/research-tools

# Load secrets
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Run the generation script
echo "=========================================="
echo "Starting controlled variation generation"
echo "=========================================="
echo ""

python probes/scripts/data_collection/generate_controlled_variation_sample.py

exit_code=$?

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code
