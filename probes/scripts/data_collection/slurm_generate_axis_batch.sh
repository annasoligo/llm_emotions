#!/bin/bash
#SBATCH --job-name=gen_axis_batch
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/gen_axis_batch_%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/gen_axis_batch_%j.err
#SBATCH --time=04:00:00
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
echo "Starting axis paraphrase batch generation"
echo "=========================================="
echo ""

python probes/scripts/data_collection/generate_axis_paraphrases_batch.py \
    --mode full \
    --n-neutral-texts 100 \
    --poll-interval 120 \
    --model claude-3-5-haiku-20241022

exit_code=$?

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code
