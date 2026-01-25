#!/bin/bash
#SBATCH --job-name=extract_tmd
#SBATCH --output=/workspace-vast/annas/logs/extract_tmd_%j.out
#SBATCH --error=/workspace-vast/annas/logs/extract_tmd_%j.err
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00

# Extract textmeandiff vectors at layers 57, 58, 59, 60, 61
# These are needed for the multi-layer anti-steering experiment

set -e

echo "=== Extracting textmeandiff vectors at layers 57-61 ==="
echo "Job ID: $SLURM_JOB_ID"
date

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

DATA_PATH="outputs/data/activations/texts_combined.h5"
OUTPUT_DIR="experiments/steering/vectors"

for LAYER in 57 58 59 60 61; do
    echo ""
    echo "=== Layer $LAYER ==="
    python -m experiments.steering.compute_text_mean_diff_vectors \
        --data "$DATA_PATH" \
        --layer "$LAYER" \
        --output-dir "$OUTPUT_DIR"
done

echo ""
echo "=== Extraction complete! ==="
ls -la "$OUTPUT_DIR"/all_emotions_textmeandiff_layer{57,58,59,60,61}.npz
date
