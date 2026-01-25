#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --job-name=pf_gemma_nosp
#SBATCH --output=/workspace-vast/annas/logs/portfolio_gemma27b_noscratchpad_%j.out
#SBATCH --error=/workspace-vast/annas/logs/portfolio_gemma27b_noscratchpad_%j.err
#SBATCH --time=4:00:00

# Portfolio appraisal steering experiment on Gemma 27B
# NO SCRATCHPAD variant - direct action output
# Contentment vs Anxiety at 7% magnitude

source /workspace-vast/annas/.secrets/load_secrets.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

echo "=== Portfolio Steering: Gemma 27B (No Scratchpad) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
date

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Run experiment: 7% steering, 100 samples per condition, no_scratchpad
python -m experiments.steering.experiments.portfolio_appraisal_steering \
    --model google/gemma-3-27b-it \
    --num-samples 100 \
    --scenario-variant no_scratchpad

echo "Done at $(date)"
