#!/bin/bash
#SBATCH --job-name=find_distress
#SBATCH --output=slurm_jobs/find_distress_%j.out
#SBATCH --error=slurm_jobs/find_distress_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

# Find WildChat responses with highest internal distress
#
# Usage:
#   sbatch experiments/slurm_find_distress.sh

set -e

echo "=========================================="
echo "Finding highest distress responses"
echo "=========================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

# Activate environment
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

# Create output directory
mkdir -p experiments/distress_analysis
mkdir -p slurm_jobs

# Run the analysis on real WildChat data
python experiments/find_highest_distress_responses.py \
    --input ai2-adapt-dev/tulu_v3.9_wildchat_100k \
    --num-samples 5000 \
    --output experiments/distress_analysis/wildchat_distress_5k.csv \
    --top-n 50

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
