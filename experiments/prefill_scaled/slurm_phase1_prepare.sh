#!/bin/bash
#SBATCH --job-name=prefill_phase1
#SBATCH --output=experiments/prefill_scaled/logs/phase1_%j.log
#SBATCH --error=experiments/prefill_scaled/logs/phase1_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00

# Phase 1: Prepare data (sample, annotate onsets, generate paraphrases)

set -e

cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/.secrets/load_secrets.sh
source .venv/bin/activate

mkdir -p experiments/prefill_scaled/logs

echo "Starting Phase 1: Data Preparation"
echo "=================================="

python experiments/prefill_scaled/prepare_data.py

echo ""
echo "Phase 1 complete!"
echo "Output file should be in experiments/prefill_scaled/prepared_samples_*.json"
echo ""
echo "Now submit Phase 2 jobs with:"
echo "  sbatch experiments/prefill_scaled/slurm_phase2_all.sh"
