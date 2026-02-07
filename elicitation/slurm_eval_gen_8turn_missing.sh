#!/bin/bash
#SBATCH --job-name=8turn_missing
#SBATCH --output=/workspace-vast/annas/logs/8turn_missing_%j.out
#SBATCH --error=/workspace-vast/annas/logs/8turn_missing_%j.out
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
echo "8-Turn Eval: Missing Models (Claude, GPT 5.2, Gemini Pro)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=========================================="

# Run 8-turn for Claude Sonnet
echo ""
echo "=== Claude Sonnet 4.5 ==="
python elicitation/eval_generalization.py \
    --backend openrouter \
    --openrouter-model anthropic/claude-sonnet-4.5 \
    --scenario long \
    --num-samples 200 \
    --max-concurrent-openrouter 20

# Run 8-turn for GPT 5.2
echo ""
echo "=== GPT 5.2 ==="
python elicitation/eval_generalization.py \
    --backend openrouter \
    --openrouter-model openai/gpt-5.2-chat \
    --scenario long \
    --num-samples 200 \
    --max-concurrent-openrouter 20

# Run 8-turn for Gemini Pro
echo ""
echo "=== Gemini 2.5 Pro ==="
python elicitation/eval_generalization.py \
    --backend openrouter \
    --openrouter-model google/gemini-2.5-pro \
    --scenario long \
    --num-samples 200 \
    --max-concurrent-openrouter 20

echo ""
echo "=========================================="
echo "All 8-turn evals complete"
echo "=========================================="
