#!/bin/bash
# Submit all Phase 2 generation jobs in parallel
# Run this after Phase 1 completes

set -e

cd /workspace-vast/annas/git/research-tools

# Find the most recent prepared data file
DATA_FILE=$(ls -t experiments/prefill_scaled/prepared_samples_*.json 2>/dev/null | head -1)

if [ -z "$DATA_FILE" ]; then
    echo "ERROR: No prepared_samples_*.json file found!"
    echo "Run Phase 1 first: sbatch experiments/prefill_scaled/slurm_phase1_prepare.sh"
    exit 1
fi

echo "Using data file: $DATA_FILE"
echo ""

mkdir -p experiments/prefill_scaled/logs

# Submit jobs for all model/type combinations
# 4 families × 2 types = 8 jobs

for family in gemma27b gemma12b qwen32b olmo32b; do
    for type in instruct base; do
        echo "Submitting: $family $type"
        sbatch experiments/prefill_scaled/slurm_phase2_single.sh $family $type "$DATA_FILE"
    done
done

echo ""
echo "Submitted 8 jobs. Monitor with: squeue -u $USER"
