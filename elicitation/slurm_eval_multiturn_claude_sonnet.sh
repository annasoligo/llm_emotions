#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --job-name=eval_mt_claude_sonnet
#SBATCH --output=/workspace-vast/annas/logs/eval_mt_claude_sonnet_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_mt_claude_sonnet_%j.err
#SBATCH --time=12:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Multiturn Frustration Eval: Claude Sonnet 4.5 (OpenRouter)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Conditions: original, variant, wildchat"
echo "Samples: 200"
echo "Thinking: N/A (Sonnet doesn't use extended thinking)"
echo "=========================================="

python -u elicitation/eval_multiturn_frustration.py \
    --backend openrouter \
    --openrouter-model anthropic/claude-sonnet-4.5 \
    --num-samples 200 \
    --num-turns 3 \
    --conditions original,variant,wildchat \
    --max-concurrent-openrouter 30 \
    --output-dir elicitation/outputs/eval_multiturn

echo "=========================================="
echo "Eval complete"
echo "=========================================="
