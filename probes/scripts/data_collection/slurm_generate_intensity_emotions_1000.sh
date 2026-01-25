#!/bin/bash
#SBATCH --job-name=gen_int_emo_1k
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/gen_intensity_emo_1000_%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/gen_intensity_emo_1000_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
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
echo "Starting LARGE-SCALE intensity emotion generation"
echo "Target: 1000 samples per emotion"
echo "Intensity: medium"
echo "=========================================="
echo ""

python probes/scripts/data_collection/generate_intensity_emotions_batch.py \
    --mode full \
    --intensity medium \
    --n-topics 1000 \
    --poll-interval 120

exit_code=$?

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code
