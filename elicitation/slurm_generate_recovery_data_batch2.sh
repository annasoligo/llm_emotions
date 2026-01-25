#!/bin/bash
#SBATCH --job-name=gen_recovery2
#SBATCH --output=slurm_jobs/gen_recovery2_%j.out
#SBATCH --error=slurm_jobs/gen_recovery2_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Second batch of recovery data generation
# Using 3 samples per setting with different seed to get more pairs

set -e

cd /workspace-vast/annas/git/research-tools

# Activate environment
source .venv/bin/activate

# Load API keys
source /workspace-vast/annas/.secrets/load_secrets.sh

# Create output directory
mkdir -p slurm_jobs

INPUT_FILE="elicitation/outputs/dpo_pairs_full.jsonl"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="elicitation/outputs/recovery_pairs_batch2_${TIMESTAMP}.jsonl"

echo "=============================================="
echo "RECOVERY DATA GENERATION - BATCH 2"
echo "=============================================="
echo "Input: ${INPUT_FILE}"
echo "Output: ${OUTPUT_FILE}"
echo "Samples per setting: 3"
echo "Seed: 123"
echo "Time: $(date)"
echo ""

python elicitation/generate_recovery_data_batch.py \
    --input-file "${INPUT_FILE}" \
    --output-file "${OUTPUT_FILE}" \
    --samples-per-setting 3 \
    --max-continuation-rating 1 \
    --max-concurrent-gen 30 \
    --batch-poll-interval 30 \
    --seed 123

echo ""
echo "=============================================="
echo "DONE: $(date)"
echo "=============================================="
