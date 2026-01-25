#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --job-name=gen_dpo_aggressive
#SBATCH --output=/workspace-vast/annas/logs/gen_dpo_aggressive_%j.out
#SBATCH --error=/workspace-vast/annas/logs/gen_dpo_aggressive_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python -u elicitation/eval_generalization.py google/gemma-3-27b-it \
    --lora-path /workspace-vast/annas/models/gemma3-27b-dpo-calm-full/2026-01-15_09-48-56 \
    --scenario aggressive \
    --num-samples 20
