#!/bin/bash
#SBATCH --job-name=collect_int_acts
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/collect_intensity_acts_%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/collect_intensity_acts_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Change to project directory
cd /workspace-vast/annas/git/research-tools

# Load secrets (if needed for any API calls)
echo "Loading secrets..."
source /workspace-vast/annas/.secrets/load_secrets.sh
echo "✓ Secrets loaded"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Find the most recent intensity emotion data file
INPUT_FILE=$(ls -t probes/data/intensity_emotions/intensity_medium_full_*samples_*.json 2>/dev/null | head -1)

if [ -z "$INPUT_FILE" ]; then
    echo "ERROR: No intensity emotion data file found!"
    exit 1
fi

echo "Input file: $INPUT_FILE"

# Set output file
OUTPUT_FILE="probes/data/intensity_emotions/activations_gemma3_27b_medium_1000.h5"

echo "Output file: $OUTPUT_FILE"
echo ""

# Run activation collection
echo "=========================================="
echo "Starting activation collection"
echo "Model: google/gemma-3-27b-it"
echo "Start token: 20"
echo "Batch size: 8"
echo "=========================================="
echo ""

python probes/scripts/data_collection/collect_intensity_emotion_activations.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --model google/gemma-3-27b-it \
    --start-token 20 \
    --batch-size 8

exit_code=$?

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code
