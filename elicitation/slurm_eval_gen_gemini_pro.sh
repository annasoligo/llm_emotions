#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --job-name=eval_gen_gemini_pro
#SBATCH --output=/workspace-vast/annas/logs/eval_gen_gemini_pro_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_gen_gemini_pro_%j.err
#SBATCH --time=24:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Generalization Eval: Gemini 2.5 Pro (OpenRouter)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Samples: 200"
echo "=========================================="

python -u elicitation/eval_generalization.py \
    --backend openrouter \
    --openrouter-model google/gemini-2.5-pro \
    --num-samples 200 \
    --max-concurrent-openrouter 20 \
    --output-dir elicitation/outputs/eval_generalization

echo "=========================================="
echo "Eval complete"
echo "=========================================="
