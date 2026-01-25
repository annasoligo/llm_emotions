#!/bin/bash
#SBATCH --job-name=base_vs_inst_test
#SBATCH --output=/workspace-vast/annas/git/research-tools/experiments/base_vs_instruct/logs/%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/experiments/base_vs_instruct/logs/%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Quick test with 2 samples

echo "========================================"
echo "BASE VS INSTRUCT TEST (2 samples)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================"

# Load secrets (for Anthropic API key and HF token)
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate venv
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Activated venv"
fi

# Set HuggingFace cache
export HF_HOME=/workspace-vast/pretrained_ckpts

# Check GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "========================================"

# Run test with 2 samples
echo "Running test with 2 samples..."
python experiments/base_vs_instruct_frustration.py --phase all --limit 2

if [ $? -ne 0 ]; then
    echo "ERROR: Test failed"
    exit 1
fi

echo "✓ Test complete"
