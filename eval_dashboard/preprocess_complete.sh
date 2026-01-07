#!/bin/bash
# Complete preprocessing wrapper for eval dashboard
# Runs all preprocessing steps in optimal order with all probe types

set -e  # Exit on error

echo "================================================================================"
echo "COMPLETE PREPROCESSING FOR EVAL DASHBOARD"
echo "================================================================================"
echo ""
echo "This script runs:"
echo "  1. Standard probe-based preprocessing (all probe types)"
echo "  2. Logit lens preprocessing (all layer ranges with baseline correction)"
echo "  3. Axis lens preprocessing (with baseline correction)"
echo ""
echo "================================================================================"

# Configuration
INPUT_FILE="${1:-/workspace-vast/annas/git/research-tools/elicitation/outputs/annotated_emotion_onset_gemma3.jsonl}"
OUTPUT_FILE="${2:-/workspace-vast/annas/git/research-tools/eval_dashboard/data/preprocessed_conversations_complete.pkl}"

echo ""
echo "Configuration:"
echo "  Input:  $INPUT_FILE"
echo "  Output: $OUTPUT_FILE"
echo ""

# Check if input exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    exit 1
fi

# Activate virtual environment
cd /workspace-vast/annas/git/research-tools
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✓ Activated virtual environment"
else
    echo "WARNING: No virtual environment found at .venv/bin/activate"
fi

echo ""
echo "================================================================================"
echo "STEP 1/3: PROBE-BASED PREPROCESSING"
echo "================================================================================"
echo ""
echo "Running data_preprocessing.py with ALL probe types:"
echo "  - orthogonal_raw"
echo "  - orthogonal_cpca_top10"
echo "  - orthogonal_regularized_lambda100"
echo "  - text_raw"
echo "  - text_cpca"
echo "  - centroid_k10"
echo "  - diverse_isolation_lambda10"
echo ""

python eval_dashboard/data_preprocessing.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --probes orthogonal_raw orthogonal_cpca_top10 orthogonal_regularized_lambda100 \
             text_raw text_cpca centroid_k10 diverse_isolation_lambda10

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Probe-based preprocessing complete"
else
    echo ""
    echo "✗ Probe-based preprocessing failed!"
    exit 1
fi

echo ""
echo "================================================================================"
echo "STEP 2/3: LOGIT LENS PREPROCESSING"
echo "================================================================================"
echo ""
echo "Running add_logit_lens_to_existing.py with:"
echo "  - All layer ranges (L40-50, L30-40, L20-30)"
echo "  - Integrated baseline correction"
echo "  - Per-layer scores for layerwise plotting"
echo ""

# Temporarily update the script to use the correct input/output paths
python eval_dashboard/add_logit_lens_to_existing.py <<EOF_PYTHON
# Override paths
import sys
from pathlib import Path
sys.path.insert(0, "/workspace-vast/annas/git/believe-it-or-not")
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import and run main with custom paths
import add_logit_lens_to_existing
add_logit_lens_to_existing.input_path = Path("$OUTPUT_FILE")
add_logit_lens_to_existing.output_path = Path("$OUTPUT_FILE")
add_logit_lens_to_existing.main()
EOF_PYTHON

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Logit lens preprocessing complete"
else
    echo ""
    echo "✗ Logit lens preprocessing failed!"
    exit 1
fi

echo ""
echo "================================================================================"
echo "STEP 3/3: AXIS LENS PREPROCESSING"
echo "================================================================================"
echo ""
echo "Running add_axis_lens_to_existing.py with:"
echo "  - Integrated baseline correction"
echo "  - Random token baseline computation"
echo ""

python eval_dashboard/add_axis_lens_to_existing.py "$OUTPUT_FILE" "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Axis lens preprocessing complete"
else
    echo ""
    echo "✗ Axis lens preprocessing failed!"
    exit 1
fi

echo ""
echo "================================================================================"
echo "PREPROCESSING COMPLETE!"
echo "================================================================================"
echo ""
echo "Output file: $OUTPUT_FILE"
echo "File size: $(du -h "$OUTPUT_FILE" | cut -f1)"
echo ""
echo "Probe types included:"
echo "  Standard probes:"
echo "    - orthogonal_raw"
echo "    - orthogonal_cpca_top10"
echo "    - orthogonal_regularized_lambda100"
echo "    - text_raw"
echo "    - text_cpca"
echo "    - centroid_k10"
echo "    - diverse_isolation_lambda10"
echo "  Logit lens:"
echo "    - logit_lens_mean (L40-50)"
echo "    - logit_lens_mean_l30_40 (L30-40)"
echo "    - logit_lens_mean_l20_30 (L20-30)"
echo "  Axis lens:"
echo "    - axis_lens_mean"
echo ""
echo "Features:"
echo "  ✓ Baseline correction applied to logit/axis lens"
echo "  ✓ Per-layer scores stored for layerwise plotting"
echo "  ✓ All probes computed in optimal order"
echo ""
echo "================================================================================"
