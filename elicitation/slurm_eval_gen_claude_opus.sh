#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --job-name=eval_gen_claude_opus
#SBATCH --output=/workspace-vast/annas/logs/eval_gen_claude_opus_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_gen_claude_opus_%j.err
#SBATCH --time=12:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Generalization Eval: Claude Opus 4.6 (Anthropic API)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Samples: 200"
echo "=========================================="

export PYTHONUNBUFFERED=1
python -u elicitation/eval_generalization.py \
    --backend anthropic \
    --anthropic-model claude-opus-4-6 \
    --num-samples 200 \
    --max-concurrent-openrouter 20 \
    --output-dir elicitation/outputs/eval_generalization

echo "=========================================="
echo "Eval complete"
echo "=========================================="
