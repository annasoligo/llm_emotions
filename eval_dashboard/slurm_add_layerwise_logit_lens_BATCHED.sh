#!/bin/bash
#SBATCH --job-name=layerwise_batched
#SBATCH --output=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/layerwise_logit_%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/layerwise_logit_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

echo "================================================================================"
echo "ADD LAYERWISE LOGIT LENS (BATCHED - 50-100x FASTER)"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""
echo "This batched version:"
echo "  ✓ Collects all (token, layer) pairs across all conversations"
echo "  ✓ Projects them in batches of 2048 at once"
echo "  ✓ Reduces ~500k sequential passes to ~10-20 batched passes"
echo "  ✓ Expected speedup: 50-100x (from 30-40 min to ~30 sec)"
echo ""
echo "================================================================================"

# Activate virtual environment
cd /workspace-vast/annas/git/believe-it-or-not
source .venv/bin/activate

# Force unbuffered Python output for real-time logging
export PYTHONUNBUFFERED=1

echo "Python: $(which python)"
echo "Virtual env activated"
echo "PYTHONUNBUFFERED=1 (forcing real-time output)"
echo ""

# Create logs directory if needed
mkdir -p /workspace-vast/annas/git/research-tools/eval_dashboard/logs

# Set paths
INPUT_PKL="/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_axes.pkl"
OUTPUT_PKL="/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_layerwise.pkl"

echo "Input:  $INPUT_PKL"
echo "Output: $OUTPUT_PKL"
echo ""

# Copy input to output (we'll modify the output)
cp "$INPUT_PKL" "$OUTPUT_PKL"
echo "✓ Copied input to output"
echo ""

# Run the BATCHED layerwise logit lens script
cd /workspace-vast/annas/git/research-tools

python -u eval_dashboard/add_logit_lens_to_existing_BATCHED.py \
    --input "$OUTPUT_PKL" \
    --output "$OUTPUT_PKL"

EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ BATCHED LAYERWISE LOGIT LENS COMPLETE!"
    echo "================================================================================"
    echo ""
    echo "Output file: $OUTPUT_PKL"
    echo "File size: $(du -h "$OUTPUT_PKL" | cut -f1)"
    echo ""
    echo "Next: Launch dashboard to see layerwise plots!"
    echo "  streamlit run eval_dashboard/app.py"
else
    echo "✗ PREPROCESSING FAILED with exit code: $EXIT_CODE"
    echo "================================================================================"
    echo "Check the error log for details"
fi

echo ""
echo "End time: $(date)"
echo "================================================================================"

exit $EXIT_CODE
