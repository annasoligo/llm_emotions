#!/bin/bash
#SBATCH --job-name=base_vs_instruct
#SBATCH --output=/workspace-vast/annas/git/research-tools/experiments/base_vs_instruct/logs/%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/experiments/base_vs_instruct/logs/%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Base vs Instruct Frustration Continuation Experiment
# Requires A100 80GB for gemma-3-27b models

echo "========================================"
echo "BASE VS INSTRUCT FRUSTRATION EXPERIMENT"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================"

# Load secrets (for Anthropic API key)
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

# Run the experiment (use --limit for testing)
LIMIT=${1:-}  # Pass limit as first argument, empty means no limit
echo "Running base vs instruct experiment..."
if [ -n "$LIMIT" ]; then
    echo "Limiting to $LIMIT samples"
    python experiments/base_vs_instruct_frustration.py --phase all --limit $LIMIT
else
    python experiments/base_vs_instruct_frustration.py --phase all
fi

if [ $? -ne 0 ]; then
    echo "ERROR: Experiment failed"
    exit 1
fi

echo "✓ Experiment complete"
