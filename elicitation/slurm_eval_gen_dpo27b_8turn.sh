#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --job-name=eval_gen_dpo_8t
#SBATCH --output=/workspace-vast/annas/logs/eval_gen_dpo27b_8turn_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_gen_dpo27b_8turn_%j.err
#SBATCH --time=08:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Generalization Eval: Gemma 27B DPO (8-turn only)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Samples: 200"
echo "=========================================="

python elicitation/eval_generalization.py \
    annasoli/gemma3-27b-dpo-calm-full-merged \
    --backend vllm \
    --num-samples 200 \
    --max-model-len 32768 \
    --skip-triggers \
    --skip-tones \
    --output-dir elicitation/outputs/eval_generalization

echo "=========================================="
echo "Eval complete"
echo "=========================================="
