#!/bin/bash
#SBATCH --job-name=trig_dpo
#SBATCH --output=/workspace-vast/annas/logs/trig_dpo_%j.out
#SBATCH --error=/workspace-vast/annas/logs/trig_dpo_%j.out
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python elicitation/eval_generalization.py \
    annasoli/gemma3-27b-dpo-calm-full-merged \
    --scenario triggers \
    --num-samples 20 \
    --tensor-parallel-size 2 \
    --max-model-len 8192
