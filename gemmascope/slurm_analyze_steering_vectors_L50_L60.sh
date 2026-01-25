#!/bin/bash
#SBATCH --job-name=analyze_sv_L50_L60
#SBATCH --output=gemmascope/slurm_jobs/analyze_steering_vectors_L50_L60_%j.out
#SBATCH --error=gemmascope/slurm_jobs/analyze_steering_vectors_L50_L60_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

# Analyze steering vectors with SAE features for layers 50 and 60

set -e

cd /workspace-vast/annas/git/research-tools

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

export HF_HOME="/workspace-vast/annas/.cache/huggingface"
export TRANSFORMERS_CACHE="/workspace-vast/annas/.cache/huggingface"
export PYTHONUNBUFFERED=1

echo "Python: $(which python)"
echo "Starting steering vector SAE analysis for layers 50 and 60..."

VECTORS_DIR="/workspace-vast/annas/git/research-tools/experiments/steering/vectors"
OUTPUT_DIR="gemmascope/outputs/steering_vector_analysis"
mkdir -p "$OUTPUT_DIR"

# Analyze layer 50
echo ""
echo "========================================="
echo " Layer 50"
echo "========================================="
python gemmascope/analyze_steering_vectors_with_sae.py \
    --vectors-path "$VECTORS_DIR/all_emotions_textmeandiff_layer50.npz" \
    --sae-layer 50 \
    --sae-width "262k" \
    --model-size "27b" \
    --use-all-layers \
    --top-k 50 \
    --output "$OUTPUT_DIR/textmeandiff_layer50_262k" \
    --load-examples

# Analyze layer 60
echo ""
echo "========================================="
echo " Layer 60"
echo "========================================="
python gemmascope/analyze_steering_vectors_with_sae.py \
    --vectors-path "$VECTORS_DIR/all_emotions_textmeandiff_layer60.npz" \
    --sae-layer 60 \
    --sae-width "262k" \
    --model-size "27b" \
    --use-all-layers \
    --top-k 50 \
    --output "$OUTPUT_DIR/textmeandiff_layer60_262k" \
    --load-examples

echo ""
echo "Done! Results saved to $OUTPUT_DIR/"
