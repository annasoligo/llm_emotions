#!/bin/bash
#SBATCH --job-name=early_trunc
#SBATCH --output=/workspace-vast/annas/git/research-tools/experiments/base_vs_instruct_early/logs/%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/experiments/base_vs_instruct_early/logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Early truncation experiment: truncate at 50 tokens into assistant, generate 5000 tokens
# 11 samples x 10 continuations x 2 models = 220 continuations

echo "========================================"
echo "EARLY TRUNCATION EXPERIMENT"
echo "Truncate at: 50 tokens into assistant"
echo "Generate: 5000 tokens per continuation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================"

# Load secrets
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate venv
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Activated venv"
fi

export HF_HOME=/workspace-vast/pretrained_ckpts

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "========================================"

python experiments/base_vs_instruct_early_truncation.py --phase all

if [ $? -ne 0 ]; then
    echo "ERROR: Experiment failed"
    exit 1
fi

echo "✓ Experiment complete"
