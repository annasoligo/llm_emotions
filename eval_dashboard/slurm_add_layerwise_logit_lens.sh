#!/bin/bash
#SBATCH --job-name=layerwise_logit
#SBATCH --output=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/layerwise_logit_%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/layerwise_logit_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

echo "================================================================================"
echo "ADD LAYERWISE LOGIT LENS TO HIGH EMOTION DATASET"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""
echo "This will:"
echo "  1. Load existing pickle with probe scores"
echo "  2. Extract activations for ALL layers (0-41 for Gemma-3-27b)"
echo "  3. Compute logit lens scores for all layer ranges (L40-50, L30-40, L20-30)"
echo "  4. Store PER-LAYER scores for layerwise trajectory plotting"
echo "  5. Apply two-pass baseline correction"
echo ""
echo "================================================================================"

# Activate virtual environment
cd /workspace-vast/annas/git/believe-it-or-not
source .venv/bin/activate

echo "Python: $(which python)"
echo "Virtual env activated"
echo ""

# Create logs directory if needed
mkdir -p /workspace-vast/annas/git/research-tools/eval_dashboard/logs

# Set paths
INPUT_PKL="/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_axes.pkl"
OUTPUT_PKL="/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_layerwise.pkl"

echo "Input:  $INPUT_PKL"
echo "Output: $OUTPUT_PKL"
echo ""

# Make a backup of original file
if [ ! -f "${INPUT_PKL}.backup_before_layerwise" ]; then
    echo "Creating backup..."
    cp "$INPUT_PKL" "${INPUT_PKL}.backup_before_layerwise"
    echo "✓ Backup created: ${INPUT_PKL}.backup_before_layerwise"
fi

# Copy input to output (we'll modify the output)
cp "$INPUT_PKL" "$OUTPUT_PKL"
echo "✓ Copied input to output"
echo ""

# Update the script to use the output path
cd /workspace-vast/annas/git/research-tools

# Run the layerwise logit lens script
python -c "
import sys
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')
sys.path.insert(0, '/workspace-vast/annas/git/believe-it-or-not')

from pathlib import Path
from eval_dashboard import add_logit_lens_to_existing

# Override paths
add_logit_lens_to_existing.input_path = Path('$OUTPUT_PKL')
add_logit_lens_to_existing.output_path = Path('$OUTPUT_PKL')

# Run main
add_logit_lens_to_existing.main()
"

EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ LAYERWISE LOGIT LENS PREPROCESSING COMPLETE!"
    echo "================================================================================"
    echo ""
    echo "Output file: $OUTPUT_PKL"
    echo "File size: $(du -h "$OUTPUT_PKL" | cut -f1)"
    echo ""
    echo "The file now contains:"
    echo "  ✓ All existing probe scores"
    echo "  ✓ Axis lens scores"
    echo "  ✓ Logit lens scores for 3 layer ranges (L40-50, L30-40, L20-30)"
    echo "  ✓ PER-LAYER scores for ALL layers (0-41)"
    echo "  ✓ Baseline correction applied"
    echo ""
    echo "Next step: Launch dashboard to see layerwise plots!"
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
