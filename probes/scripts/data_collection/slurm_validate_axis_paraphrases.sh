#!/bin/bash
#SBATCH --job-name=validate_axis
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/validate_axis_%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/validate_axis_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Change to project directory
cd /workspace-vast/annas/git/research-tools

# Load secrets (not needed for validation but keeping consistent)
echo "Loading secrets..."
source /workspace-vast/annas/.secrets/load_secrets.sh
echo "✓ Secrets loaded"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Run the validation script
echo "=========================================="
echo "Starting axis paraphrase validation"
echo "=========================================="
echo ""

python probes/scripts/data_collection/validate_axis_paraphrases.py \
    --input probes/data/axis_paraphrases/axis_paraphrases_test.jsonl \
    --layers 40-50 \
    --device cuda

exit_code=$?

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code
