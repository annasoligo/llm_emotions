#!/bin/bash
#SBATCH --job-name=judge_sandbag
#SBATCH --output=/workspace-vast/annas/logs/judge_sandbag_%j.out
#SBATCH --error=/workspace-vast/annas/logs/judge_sandbag_%j.err
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00

# Run sandbagging + coherency judges on all generation outputs
# Uses Anthropic Batch API for efficiency
#
# Usage:
#   sbatch slurm_judge_sandbagging_full.sh [pattern]
#
# Examples:
#   sbatch slurm_judge_sandbagging_full.sh  # judges all recent sandbagging_gemma*.jsonl
#   sbatch slurm_judge_sandbagging_full.sh "sandbagging_gemma*_20260121*.jsonl"

set -e

# Load secrets for Anthropic API
source /workspace-vast/annas/.secrets/load_secrets.sh

echo "=== Sandbagging Judge Batch ==="
echo "Job ID: $SLURM_JOB_ID"
date

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

OUTPUT_DIR="experiments/steering/outputs"

# Pattern from argument or default
PATTERN="${1:-sandbagging_gemma*.jsonl}"

echo "Looking for files matching: $OUTPUT_DIR/$PATTERN"

# Find files that need judging (not already .judged.jsonl)
FILES=$(find "$OUTPUT_DIR" -name "$PATTERN" ! -name "*.judged.jsonl" -type f | sort)

if [ -z "$FILES" ]; then
    echo "No files found matching pattern"
    exit 0
fi

echo "Found files to judge:"
echo "$FILES"
echo ""

# Process each file
for f in $FILES; do
    # Skip answer_only format (no scratchpad to judge)
    if [[ "$f" == *"answer_only"* ]]; then
        echo "Skipping answer_only format: $f"
        continue
    fi

    echo "=========================================="
    echo "Judging: $f"
    echo "=========================================="

    python -m experiments.steering.experiments.judge_sandbagging_coherency_batch \
        --input "$f" \
        --poll-interval 30

    echo ""
done

echo "All judging complete!"
date
