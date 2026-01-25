#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=preproc_baseline_v12
#SBATCH --output=/workspace-vast/annas/logs/preproc_baseline_v12_%j.out
#SBATCH --error=/workspace-vast/annas/logs/preproc_baseline_v12_%j.err
#SBATCH --time=02:00:00

# Preprocess baseline V12 solvable responses for dashboard
# Applies all 5 probe types used in other subsets

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

cd eval_dashboard

echo "=========================================="
echo "Baseline V12 Preprocessing"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

INPUT_FILE="/workspace-vast/annas/git/research-tools/elicitation/outputs/baseline_v12_for_preprocessing.jsonl"
OUTPUT_FILE="data/baseline_v12_solvable.pkl"

echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo ""

# Run preprocessing with all 5 probe types
python data_preprocessing.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --probes orthogonal_raw orthogonal_cpca_top10 text_raw text_cpca centroid_k10

echo ""
echo "=========================================="
echo "Preprocessing complete"
echo "=========================================="

# Show output file
if [ -f "$OUTPUT_FILE" ]; then
    echo "✓ Output file: $OUTPUT_FILE"
    echo "  Size: $(du -h "$OUTPUT_FILE" | cut -f1)"
else
    echo "✗ Output file not found"
    exit 1
fi

# Create compressed version
echo ""
echo "Creating compressed version..."
gzip -c "$OUTPUT_FILE" > "${OUTPUT_FILE}.gz"
echo "✓ Compressed: ${OUTPUT_FILE}.gz"
echo "  Size: $(du -h "${OUTPUT_FILE}.gz" | cut -f1)"
