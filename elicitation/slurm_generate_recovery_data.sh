#!/bin/bash
#SBATCH --job-name=gen_recovery
#SBATCH --output=slurm_jobs/gen_recovery_%j.out
#SBATCH --error=slurm_jobs/gen_recovery_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Generate recovery training data for DPO using batch API
# Creates 2 samples per (pair, recovery_point) setting for redundancy

set -e

cd /workspace-vast/annas/git/research-tools

# Activate environment
source .venv/bin/activate

# Load API keys
source /workspace-vast/annas/.secrets/load_secrets.sh

# Create output directory
mkdir -p slurm_jobs

# Input: existing DPO pairs (uses 'rejected' field as frustrated responses)
INPUT_FILE="elicitation/outputs/dpo_pairs_full.jsonl"

# Output: recovery pairs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="elicitation/outputs/recovery_pairs_${TIMESTAMP}.jsonl"

echo "=============================================="
echo "RECOVERY DATA GENERATION (BATCH API)"
echo "=============================================="
echo "Input: ${INPUT_FILE}"
echo "Output: ${OUTPUT_FILE}"
echo "Samples per setting: 2"
echo "Time: $(date)"
echo ""

python elicitation/generate_recovery_data_batch.py \
    --input-file "${INPUT_FILE}" \
    --output-file "${OUTPUT_FILE}" \
    --samples-per-setting 2 \
    --max-continuation-rating 1 \
    --max-concurrent-gen 30 \
    --batch-poll-interval 30

echo ""
echo "=============================================="
echo "DONE: $(date)"
echo "=============================================="
