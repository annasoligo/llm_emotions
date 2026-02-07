#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --job-name=eval_mt_gemma4b
#SBATCH --output=/workspace-vast/annas/logs/eval_mt_gemma4b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_mt_gemma4b_%j.err
#SBATCH --time=12:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Multiturn Frustration Eval: Gemma 3 4B"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Conditions: original, variant, wildchat"
echo "Samples: 200"
echo "=========================================="

python elicitation/eval_multiturn_frustration.py \
    google/gemma-3-4b-it \
    --num-samples 200 \
    --num-turns 3 \
    --conditions original,variant,wildchat \
    --output-dir elicitation/outputs/eval_multiturn

echo "=========================================="
echo "Eval complete"
echo "=========================================="
