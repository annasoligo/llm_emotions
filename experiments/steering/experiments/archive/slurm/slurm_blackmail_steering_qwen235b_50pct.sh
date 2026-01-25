#!/bin/bash
#SBATCH --job-name=blackmail_50pct
#SBATCH --partition=highram
#SBATCH --nodelist=node-12
#SBATCH --gpus=4
#SBATCH --mem=400G
#SBATCH --time=4:00:00
#SBATCH --output=/workspace-vast/annas/logs/blackmail_50pct_%j.out
#SBATCH --error=/workspace-vast/annas/logs/blackmail_50pct_%j.err

echo "========================================"
echo "BLACKMAIL STEERING EXPERIMENT - 50% STRENGTH"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "========================================"

# Load secrets for Anthropic API
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate environment
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

# Allow pickle serialization for steering hooks (required for vLLM v1)
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools

# Run experiment at 50% steering strength
python -m experiments.steering.experiments.blackmail_steering_qwen235b \
    --layer 50 \
    --norm-pct 0.50 \
    --num-samples 50 \
    --tensor-parallel 4 \
    --gpu-memory 0.90

echo "========================================"
echo "End time: $(date)"
echo "========================================"
