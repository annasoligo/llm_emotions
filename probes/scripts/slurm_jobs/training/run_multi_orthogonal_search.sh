#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=multi_ortho_search
#SBATCH --output=/workspace-vast/annas/logs/%j.out
#SBATCH --error=/workspace-vast/annas/logs/%j.err

# Search for optimal K for multi-orthogonal emotion probes

# Load environment
source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME=/workspace-vast/pretrained_ckpts

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Configuration
LAYER=${1:-30}  # Default layer 30
MAX_K=${2:-8}   # Search up to K=8
ORTHO_WEIGHT=${3:-10.0}

echo "========================================"
echo "Multi-Orthogonal Probe K-Search"
echo "========================================"
echo "Layer: $LAYER"
echo "Max K: $MAX_K"
echo "Ortho Weight: $ORTHO_WEIGHT"
echo

# Run search
python probes/scripts/training/train_multi_orthogonal_text_probes.py \
    --layer $LAYER \
    --search-k \
    --max-sets $MAX_K \
    --ortho-weight $ORTHO_WEIGHT \
    --max-epochs 200 \
    --patience 20 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --convergence-threshold 0.1 \
    --min-accuracy-threshold 0.25 \
    --accuracy-degradation-threshold 0.05 \
    --device cuda

echo
echo "========================================"
echo "Search Complete!"
echo "========================================"
