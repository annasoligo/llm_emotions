#!/bin/bash
#SBATCH --job-name=analyze_sv
#SBATCH --output=gemmascope/slurm_jobs/analyze_steering_vectors_%j.out
#SBATCH --error=gemmascope/slurm_jobs/analyze_steering_vectors_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

# Analyze steering vectors with SAE features

set -e

cd /workspace-vast/annas/git/research-tools

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

export HF_HOME="/workspace-vast/annas/.cache/huggingface"
export TRANSFORMERS_CACHE="/workspace-vast/annas/.cache/huggingface"
export PYTHONUNBUFFERED=1

echo "Python: $(which python)"
echo "Starting steering vector SAE analysis..."

VECTORS_DIR="/workspace-vast/annas/git/research-tools/experiments/steering/vectors"
OUTPUT_DIR="gemmascope/outputs/steering_vector_analysis"
mkdir -p "$OUTPUT_DIR"

# Analyze layer 20 vectors with layer 20 SAE
echo ""
echo "========================================="
echo " Layer 20"
echo "========================================="
python gemmascope/analyze_steering_vectors_with_sae.py \
    --vectors-path "$VECTORS_DIR/all_emotions_textmeandiff_layer20.npz" \
    --sae-layer 20 \
    --sae-width "262k" \
    --model-size "27b" \
    --use-all-layers \
    --top-k 50 \
    --output "$OUTPUT_DIR/textmeandiff_layer20_262k" \
    --load-examples

# Analyze layer 40 vectors with layer 40 SAE (standard layer)
echo ""
echo "========================================="
echo " Layer 40"
echo "========================================="
python gemmascope/analyze_steering_vectors_with_sae.py \
    --vectors-path "$VECTORS_DIR/all_emotions_textmeandiff_layer40.npz" \
    --sae-layer 40 \
    --sae-width "262k" \
    --model-size "27b" \
    --use-all-layers \
    --top-k 50 \
    --output "$OUTPUT_DIR/textmeandiff_layer40_262k" \
    --load-examples

# Analyze layer 30 vectors with layer 30 SAE (all layers has it)
echo ""
echo "========================================="
echo " Layer 30"
echo "========================================="
python gemmascope/analyze_steering_vectors_with_sae.py \
    --vectors-path "$VECTORS_DIR/all_emotions_textmeandiff_layer30.npz" \
    --sae-layer 30 \
    --sae-width "262k" \
    --model-size "27b" \
    --use-all-layers \
    --top-k 50 \
    --output "$OUTPUT_DIR/textmeandiff_layer30_262k" \
    --load-examples

echo ""
echo "Done! Results saved to $OUTPUT_DIR/"
