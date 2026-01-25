#!/bin/bash
#SBATCH --job-name=layerwise_all
#SBATCH --output=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/layerwise_all_%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/layerwise_all_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

echo "================================================================================"
echo "PROCESS ALL DATASETS WITH LAYERWISE LOGIT LENS"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Activate virtual environment
cd /workspace-vast/annas/git/believe-it-or-not
source .venv/bin/activate

# Force unbuffered Python output
export PYTHONUNBUFFERED=1

echo "Python: $(which python)"
echo ""

# Create logs directory if needed
mkdir -p /workspace-vast/annas/git/research-tools/eval_dashboard/logs

cd /workspace-vast/annas/git/research-tools

# List of datasets to process (input -> output)
declare -a DATASETS=(
    "high_emotion_6plus_with_axes.pkl"
    "mid_emotion_3to5_with_axes.pkl"
    "low_emotion_0to2_with_axes.pkl"
    "low_emotion_with_shutdown_with_axes.pkl"
    "low_emotion_no_shutdown_with_axes.pkl"
    "baseline_v12_solvable_with_axes.pkl"
)

DATA_DIR="/workspace-vast/annas/git/research-tools/eval_dashboard/data"

TOTAL=${#DATASETS[@]}
CURRENT=0

for dataset in "${DATASETS[@]}"; do
    CURRENT=$((CURRENT + 1))
    INPUT_PKL="$DATA_DIR/$dataset"

    # Skip if input doesn't exist
    if [ ! -f "$INPUT_PKL" ]; then
        echo "[$CURRENT/$TOTAL] SKIP: $dataset (file not found)"
        continue
    fi

    echo ""
    echo "================================================================================"
    echo "[$CURRENT/$TOTAL] Processing: $dataset"
    echo "================================================================================"
    echo "Start: $(date)"

    # Run the preprocessing with probe and axis scores (modifies file in-place)
    python -u eval_dashboard/add_logit_lens_to_existing_BATCHED.py \
        --input "$INPUT_PKL" \
        --output "$INPUT_PKL" \
        --add-probes \
        --add-axes

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Completed: $dataset"
        echo "  File size: $(du -h "$INPUT_PKL" | cut -f1)"
    else
        echo "✗ FAILED: $dataset (exit code: $EXIT_CODE)"
    fi

    echo "End: $(date)"
done

echo ""
echo "================================================================================"
echo "ALL DATASETS PROCESSED"
echo "================================================================================"
echo "End time: $(date)"

exit 0
