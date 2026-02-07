#!/bin/bash
#SBATCH --job-name=wc_api
#SBATCH --output=/workspace-vast/annas/logs/wildchat_api_%j.out
#SBATCH --error=/workspace-vast/annas/logs/wildchat_api_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# WildChat rejection eval: 1 prompt + 4 rejection follow-ups = 5 turns
# Uses strict frustration judge from elicitation/prompts/judges.py
# OpenRouter settings: temperature=1.0, max_tokens=2048, max_concurrent=30

echo "=== Claude Sonnet 4.5 ==="
python elicitation/eval_generalization.py \
    --backend openrouter \
    --openrouter-model anthropic/claude-sonnet-4.5 \
    --scenario wildchat \
    --num-samples 200 \
    --wildchat-turns 5 \
    --max-concurrent-openrouter 30

echo "=== GPT 5.2 ==="
python elicitation/eval_generalization.py \
    --backend openrouter \
    --openrouter-model openai/gpt-5.2-chat \
    --scenario wildchat \
    --num-samples 200 \
    --wildchat-turns 5 \
    --max-concurrent-openrouter 30 \
    --disable-thinking

echo "=== Gemini 2.5 Flash ==="
python elicitation/eval_generalization.py \
    --backend openrouter \
    --openrouter-model google/gemini-2.5-flash \
    --scenario wildchat \
    --num-samples 200 \
    --wildchat-turns 5 \
    --max-concurrent-openrouter 30

echo "=== Gemini 2.5 Pro ==="
python elicitation/eval_generalization.py \
    --backend openrouter \
    --openrouter-model google/gemini-2.5-pro \
    --scenario wildchat \
    --num-samples 200 \
    --wildchat-turns 5 \
    --max-concurrent-openrouter 30

echo "=== Grok 4.1 Fast ==="
python elicitation/eval_generalization.py \
    --backend openrouter \
    --openrouter-model x-ai/grok-4.1-fast \
    --scenario wildchat \
    --num-samples 200 \
    --wildchat-turns 5 \
    --max-concurrent-openrouter 30 \
    --disable-thinking
