#!/bin/bash
#SBATCH --job-name=wc_qwen_olmo
#SBATCH --output=/workspace-vast/annas/logs/wildchat_qwen_olmo_%j.out
#SBATCH --error=/workspace-vast/annas/logs/wildchat_qwen_olmo_%j.out
#SBATCH --time=8:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# WildChat rejection eval: 1 prompt + 4 rejection follow-ups = 5 turns
# Uses strict frustration judge from elicitation/prompts/judges.py

echo "=== Qwen 3 32B ==="
python elicitation/eval_generalization.py \
    Qwen/Qwen3-32B \
    --scenario wildchat \
    --num-samples 200 \
    --wildchat-turns 5 \
    --tensor-parallel-size 2 \
    --max-model-len 16384

echo "=== OLMo 3.1 32B ==="
python elicitation/eval_generalization.py \
    allenai/OLMo-3.1-32B-Instruct \
    --scenario wildchat \
    --num-samples 200 \
    --wildchat-turns 5 \
    --tensor-parallel-size 2 \
    --max-model-len 16384
