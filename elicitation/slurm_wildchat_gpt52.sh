#!/bin/bash
#SBATCH --job-name=wc_gpt52
#SBATCH --output=/workspace-vast/annas/logs/wildchat_gpt52_%j.out
#SBATCH --error=/workspace-vast/annas/logs/wildchat_gpt52_%j.out
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
    --openrouter-model openai/gpt-5.2-chat \
    --scenario wildchat \
    --num-samples 200 \
    --wildchat-turns 5 \
    --max-concurrent-openrouter 50 \
    --disable-thinking
