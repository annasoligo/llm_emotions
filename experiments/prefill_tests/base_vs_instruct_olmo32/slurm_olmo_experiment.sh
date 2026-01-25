#!/bin/bash
#SBATCH --job-name=olmo_frust
#SBATCH --output=/workspace-vast/annas/git/research-tools/experiments/base_vs_instruct_olmo32/logs/%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/experiments/base_vs_instruct_olmo32/logs/%j.err
#SBATCH --time=16:00:00
#SBATCH --mem=140G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# OLMo 32B base vs instruct frustration experiment
# Both early (50 tokens) and late (onset) truncation

echo "========================================"
echo "OLMO FRUSTRATION EXPERIMENT"
echo "Models: allenai/OLMo-3.1-32B-Instruct, allenai/OLMo-3-1125-32B"
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
    --model-family olmo32 \
    --phase all \
    --truncation both \
    --model-type both

if [ $? -ne 0 ]; then
    echo "ERROR: Experiment failed"
    exit 1
fi

echo "✓ OLMo experiment complete"
