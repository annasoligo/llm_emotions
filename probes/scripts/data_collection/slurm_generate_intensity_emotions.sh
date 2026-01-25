#!/bin/bash
#SBATCH --job-name=gen_intensity_emo
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/gen_intensity_emo_%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/gen_intensity_emo_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

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
echo "Loading secrets..."
source /workspace-vast/annas/.secrets/load_secrets.sh
echo "✓ Secrets loaded"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Run the batch generation script
echo "=========================================="
echo "Starting intensity emotion batch generation"
echo "Mode: test (3 topics)"
echo "Intensity: medium"
echo "=========================================="
echo ""

python probes/scripts/data_collection/generate_intensity_emotions_batch.py \
    --mode test \
    --intensity medium \
    --poll-interval 30

exit_code=$?

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code
