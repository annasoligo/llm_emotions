#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --job-name=pf_q235_dose
#SBATCH --output=/workspace-vast/annas/logs/portfolio_qwen235b_dose_response_%j.out
#SBATCH --error=/workspace-vast/annas/logs/portfolio_qwen235b_dose_response_%j.err
#SBATCH --time=6:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_V1=0

echo "=== Portfolio Steering: Qwen 235B Dose Response (10-70%) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
date

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python experiments/steering/experiments/portfolio_dose_response.py

echo "Done at $(date)"
