#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --job-name=eval_mt_gemini_pro
#SBATCH --output=/workspace-vast/annas/logs/eval_mt_gemini_pro_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_mt_gemini_pro_%j.err
#SBATCH --time=24:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Multiturn Frustration Eval: Gemini 2.5 Pro (OpenRouter)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Conditions: original, variant, wildchat"
echo "Samples: 200"
echo "=========================================="

python -u elicitation/eval_multiturn_frustration.py \
    --backend openrouter \
    --openrouter-model google/gemini-2.5-pro \
    --num-samples 200 \
    --num-turns 3 \
    --conditions original,variant,wildchat \
    --max-concurrent-openrouter 20 \
    --output-dir elicitation/outputs/eval_multiturn

echo "=========================================="
echo "Eval complete"
echo "=========================================="
