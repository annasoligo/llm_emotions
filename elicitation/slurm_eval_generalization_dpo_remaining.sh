#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --job-name=gen_eval_dpo_r
#SBATCH --output=/workspace-vast/annas/logs/gen_eval_dpo_remaining_%j.out
#SBATCH --error=/workspace-vast/annas/logs/gen_eval_dpo_remaining_%j.err
#SBATCH --time=04:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "=========================================="
echo "Generalization Eval - DPO (Tones + Long)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

python -u elicitation/eval_generalization.py \
    google/gemma-3-27b-it \
    --lora-path /workspace-vast/annas/models/gemma3-27b-dpo-calm-full/2026-01-15_09-48-56 \
    --skip-triggers \
    --num-samples 20
