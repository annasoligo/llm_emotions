#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --job-name=pf_gemma
#SBATCH --output=/workspace-vast/annas/logs/portfolio_gemma_%j.out
#SBATCH --error=/workspace-vast/annas/logs/portfolio_gemma_%j.err
#SBATCH --time=4:00:00

# Portfolio appraisal steering experiment on Gemma 3 27B
# Contentment vs Anxiety at 7% magnitude

source /workspace-vast/annas/.secrets/load_secrets.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

echo "=== Portfolio Steering: Gemma 3 27B ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
date

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Run experiment: 7% steering, 100 samples per condition
# Conditions: baseline, contentment (+valence -uncertainty), anxiety (-valence +uncertainty)
python -m experiments.steering.experiments.portfolio_appraisal_steering \
    --model google/gemma-3-27b-it \
    --layer 30 \
    --norm-pct 0.07 \
    --num-samples 100 \
    --max-tokens 4000 \
    --gpu-memory 0.90

echo "Done at $(date)"
