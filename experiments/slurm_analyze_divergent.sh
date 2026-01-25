#!/bin/bash
#SBATCH --job-name=analyze_divergent
#SBATCH --output=slurm_jobs/analyze_divergent_%j.out
#SBATCH --error=slurm_jobs/analyze_divergent_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

# Analyze divergent probe/logit cases with per-layer breakdown
#
# Usage:
#   sbatch experiments/slurm_analyze_divergent.sh

set -e

echo "=========================================="
echo "Analyzing divergent cases (layerwise)"
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

# Run the analysis on 5k divergent cases with ALL layers
python experiments/analyze_divergent_layerwise.py \
    --divergent-csv experiments/distress_analysis/divergent_scores_5k.csv \
    --wildchat-csv experiments/distress_analysis/wildchat_distress_5k.csv \
    --output experiments/distress_analysis/divergent_layerwise_all.csv \
    --output-txt experiments/distress_analysis/divergent_analysis_all.txt

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
