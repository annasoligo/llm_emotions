#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --job-name=eval_gen_gemma27b
#SBATCH --output=/workspace-vast/annas/logs/eval_gen_gemma27b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_gen_gemma27b_%j.err
#SBATCH --time=24:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Generalization Eval: Gemma 3 27B"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Samples: 200"
echo "=========================================="

python elicitation/eval_generalization.py \
    google/gemma-3-27b-it \
    --backend vllm \
    --num-samples 200 \
    --max-model-len 32768 \
    --output-dir elicitation/outputs/eval_generalization

echo "=========================================="
echo "Eval complete"
echo "=========================================="
