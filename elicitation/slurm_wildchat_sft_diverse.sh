#!/bin/bash
#SBATCH --job-name=wc_sft_div
#SBATCH --output=/workspace-vast/annas/logs/wildchat_sft_diverse_%j.out
#SBATCH --error=/workspace-vast/annas/logs/wildchat_sft_diverse_%j.out
#SBATCH --time=4:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --exclude=node-2,node-11,node-30

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export PYTHONUNBUFFERED=1

python elicitation/eval_generalization.py \
    google/gemma-3-27b-it \
    --lora-path /workspace-vast/annas/models/gemma3-27b-lowfrust-diverse-calm/2026-01-15_08-59-28 \
    --scenario wildchat \
    --num-samples 40 \
    --wildchat-turns 5 \
    --tensor-parallel-size 2 \
    --max-model-len 16384
