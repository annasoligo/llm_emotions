#!/bin/bash
#SBATCH --job-name=qwen_frust
#SBATCH --output=/workspace-vast/annas/git/research-tools/experiments/base_vs_instruct_qwen/logs/%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/experiments/base_vs_instruct_qwen/logs/%j.err
#SBATCH --time=16:00:00
#SBATCH --mem=140G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Qwen2.5-32B base vs instruct frustration experiment
# Both early (50 tokens) and late (onset) truncation

echo "========================================"
echo "QWEN FRUSTRATION EXPERIMENT"
echo "Models: Qwen/Qwen2.5-32B-Instruct, Qwen/Qwen2.5-32B"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================"

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Activated venv"
fi

export HF_HOME=/workspace-vast/pretrained_ckpts

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "========================================"

python experiments/base_vs_instruct_multimodel.py \
    --model-family qwen \
    --phase all \
    --truncation both \
    --model-type both

if [ $? -ne 0 ]; then
    echo "ERROR: Experiment failed"
    exit 1
fi

echo "✓ Qwen experiment complete"
