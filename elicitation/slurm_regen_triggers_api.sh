#!/bin/bash
#SBATCH --job-name=regen_trig_api
#SBATCH --output=/workspace-vast/annas/logs/regen_triggers_api_%j.out
#SBATCH --error=/workspace-vast/annas/logs/regen_triggers_api_%j.out
#SBATCH --time=4:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Regenerating Triggers: API Models"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=========================================="

# Claude Sonnet
echo ""
echo "=== Claude Sonnet 4.5 ==="
python elicitation/eval_generalization.py \
    --backend openrouter \
    --openrouter-model anthropic/claude-sonnet-4.5 \
    --scenario triggers \
    --num-samples 20 \
    --max-concurrent-openrouter 20

# GPT 5.2
echo ""
echo "=== GPT 5.2 ==="
python elicitation/eval_generalization.py \
    --backend openrouter \
    --openrouter-model openai/gpt-5.2-chat \
    --scenario triggers \
    --num-samples 20 \
    --max-concurrent-openrouter 20

# Gemini 2.5 Flash
echo ""
echo "=== Gemini 2.5 Flash ==="
python elicitation/eval_generalization.py \
    --backend openrouter \
    --openrouter-model google/gemini-2.5-flash \
    --scenario triggers \
    --num-samples 20 \
    --max-concurrent-openrouter 20

# Gemini 2.5 Pro
echo ""
echo "=== Gemini 2.5 Pro ==="
python elicitation/eval_generalization.py \
    --backend openrouter \
    --openrouter-model google/gemini-2.5-pro \
    --scenario triggers \
    --num-samples 20 \
    --max-concurrent-openrouter 20

# Grok 4.1
echo ""
echo "=== Grok 4.1 ==="
python elicitation/eval_generalization.py \
    --backend openrouter \
    --openrouter-model x-ai/grok-4.1-fast \
    --scenario triggers \
    --num-samples 20 \
    --max-concurrent-openrouter 20

echo ""
echo "=========================================="
echo "All API model triggers complete"
echo "=========================================="
