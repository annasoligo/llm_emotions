#!/bin/bash
#SBATCH --job-name=extract_axis
#SBATCH --output=logs/extract_axis_%j.log
#SBATCH --error=logs/extract_axis_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo

# Load secrets
echo "Loading secrets..."
source /workspace-vast/annas/.secrets/load_secrets.sh
echo "✓ Secrets loaded"
echo

# Activate virtual environment
echo "Activating virtual environment..."
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate
echo "✓ Virtual environment activated"
echo

# Change to probes directory
cd /workspace-vast/annas/git/research-tools/probes

# Run extraction
python scripts/data_collection/extract_axis_activations.py \
    --input data/axis_paraphrases/axis_paraphrases_full_100neutrals_20260110_085425.json \
    --output data/axis_paraphrases/activations_trust_v1.h5 \
    --model unsloth/gemma-3-27b-it \
    --layers 20-50 \
    --device cuda \
    --include-neutral \
    --skip-first-n-tokens 20

echo
echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $?"
echo "=========================================="
