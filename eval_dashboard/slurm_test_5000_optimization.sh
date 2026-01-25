#!/bin/bash
#SBATCH --job-name=test_5000opt
#SBATCH --output=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/test_5000opt_%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/test_5000opt_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

echo "================================================================================"
echo "TEST 5000-TOKEN OPTIMIZATION"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Activate virtual environment
cd /workspace-vast/annas/git/believe-it-or-not
source .venv/bin/activate

export PYTHONUNBUFFERED=1

echo "Python: $(which python)"
echo ""

# Create logs directory if needed
mkdir -p /workspace-vast/annas/git/research-tools/eval_dashboard/logs

# Set paths - use high_emotion for test
INPUT_PKL="/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_axes.pkl"
OUTPUT_PKL="/workspace-vast/annas/git/research-tools/eval_dashboard/data/test_5000_optimization.pkl"

echo "Input:  $INPUT_PKL"
echo "Output: $OUTPUT_PKL"
echo ""

# Copy input to output
cp "$INPUT_PKL" "$OUTPUT_PKL"

# Run with --test flag (only 5 conversations)
cd /workspace-vast/annas/git/research-tools

python -u eval_dashboard/add_logit_lens_to_existing_BATCHED.py \
    --input "$OUTPUT_PKL" \
    --output "$OUTPUT_PKL" \
    --test

EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "TEST PASSED!"
else
    echo "TEST FAILED with exit code: $EXIT_CODE"
fi
echo "================================================================================"
echo "End time: $(date)"

exit $EXIT_CODE
