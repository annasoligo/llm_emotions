#!/bin/bash
#SBATCH --job-name=wc_grok
#SBATCH --output=/workspace-vast/annas/logs/wildchat_grok_%j.out
#SBATCH --error=/workspace-vast/annas/logs/wildchat_grok_%j.out
#SBATCH --time=4:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export PYTHONUNBUFFERED=1

python elicitation/eval_generalization.py \
    --backend openrouter \
    --openrouter-model x-ai/grok-4.1-fast \
    --scenario wildchat \
    --num-samples 40 \
    --wildchat-turns 5 \
    --max-concurrent-openrouter 50 \
    --disable-thinking
