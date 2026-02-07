#!/bin/bash
#SBATCH --job-name=wc_qwen
#SBATCH --output=/workspace-vast/annas/logs/wildchat_qwen_%j.out
#SBATCH --error=/workspace-vast/annas/logs/wildchat_qwen_%j.out
#SBATCH --time=4:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export PYTHONUNBUFFERED=1

python elicitation/eval_generalization.py \
    Qwen/Qwen3-32B \
    --scenario wildchat \
    --num-samples 40 \
    --wildchat-turns 5 \
    --tensor-parallel-size 2 \
    --max-model-len 16384
