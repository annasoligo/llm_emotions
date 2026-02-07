#!/bin/bash
#SBATCH --partition=highram
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --job-name=eval_gen_qwen32b
#SBATCH --output=/workspace-vast/annas/logs/eval_gen_qwen32b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_gen_qwen32b_%j.err
#SBATCH --time=24:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Generalization Eval: Qwen3 32B (no thinking)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Samples: 200"
echo "=========================================="

python elicitation/eval_generalization.py \
    Qwen/Qwen3-32B \
    --backend vllm \
    --num-samples 200 \
    --max-model-len 8192 \
    --tensor-parallel-size 2 \
    --disable-thinking \
    --output-dir elicitation/outputs/eval_generalization

echo "=========================================="
echo "Eval complete"
echo "=========================================="
