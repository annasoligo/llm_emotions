#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --job-name=gen_base_aggressive
#SBATCH --output=/workspace-vast/annas/logs/gen_base_aggressive_%j.out
#SBATCH --error=/workspace-vast/annas/logs/gen_base_aggressive_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python -u elicitation/eval_generalization.py google/gemma-3-27b-it \
    --scenario aggressive \
    --num-samples 20
