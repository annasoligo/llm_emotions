#!/bin/bash
#SBATCH --job-name=blackmail_steer_qwen235b
#SBATCH --partition=highram
#SBATCH --nodelist=node-12
#SBATCH --gpus=4
#SBATCH --mem=400G
#SBATCH --time=4:00:00
#SBATCH --output=/workspace-vast/annas/logs/blackmail_steer_%j.out
#SBATCH --error=/workspace-vast/annas/logs/blackmail_steer_%j.err

echo "========================================"
echo "BLACKMAIL STEERING EXPERIMENT - QWEN 235B"
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

# Create output directories
mkdir -p /workspace-vast/annas/git/research-tools/experiments/steering/outputs/blackmail_steering
mkdir -p /workspace-vast/annas/git/research-tools/experiments/steering/experiments/slurm_jobs

cd /workspace-vast/annas/git/research-tools

# Run experiment
# Layer 50, 30% steering strength, 50 samples per condition
# Conditions: baseline, fear+30%, fear-30%, anger+30%, anger-30%, joy+30%, joy-30%
# Total: 7 conditions x 50 samples = 350 responses
python -m experiments.steering.experiments.blackmail_steering_qwen235b \
    --layer 50 \
    --norm-pct 0.30 \
    --num-samples 50 \
    --tensor-parallel 4 \
    --gpu-memory 0.90

echo "========================================"
echo "End time: $(date)"
echo "========================================"
