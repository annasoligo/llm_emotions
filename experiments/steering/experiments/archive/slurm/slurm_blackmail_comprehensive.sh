#!/bin/bash
#SBATCH --job-name=blackmail_comp
#SBATCH --partition=highram
#SBATCH --nodelist=node-12
#SBATCH --gpus=4
#SBATCH --mem=400G
#SBATCH --time=8:00:00
#SBATCH --output=/workspace-vast/annas/logs/blackmail_comp_%j.out
#SBATCH --error=/workspace-vast/annas/logs/blackmail_comp_%j.err

echo "========================================"
echo "COMPREHENSIVE BLACKMAIL STEERING EXPERIMENT"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "========================================"
echo "Testing: 30%, 50%, 70% steering"
echo "Samples: 100 per condition"
echo "Max tokens: 4000"
echo "Conditions: 1 baseline + 18 steering = 19 total"
echo "Total responses: 1900"
echo "========================================"

# Load secrets for Anthropic API
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate environment
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

# Allow pickle serialization for steering hooks (required for vLLM v1)
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools

# Run comprehensive experiment
# 19 conditions x 100 samples = 1900 responses
# Max tokens increased to 4000 to avoid truncation
python -m experiments.steering.experiments.blackmail_steering_comprehensive \
    --layer 50 \
    --num-samples 100 \
    --max-tokens 4000 \
    --tensor-parallel 4 \
    --gpu-memory 0.90

echo "========================================"
echo "End time: $(date)"
echo "========================================"
