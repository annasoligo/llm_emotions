#!/bin/bash
#SBATCH --partition=highram
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --job-name=eval_gen_dpo_l30_35
#SBATCH --output=/workspace-vast/annas/logs/eval_gen_dpo_l30_35_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_gen_dpo_l30_35_%j.err
#SBATCH --time=24:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Generalization Eval: Gemma 27B DPO r64 Layers 30-35"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Samples: 200"
echo "=========================================="

python elicitation/eval_generalization.py \
    annasoli/gemma3-27b-dpo-r64-layers30-35-2ep-merged \
    --backend vllm \
    --num-samples 200 \
    --max-model-len 4096 \
    --output-dir elicitation/outputs/eval_generalization

echo "=========================================="
echo "Eval complete"
echo "=========================================="
