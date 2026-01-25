#!/bin/bash
#SBATCH --job-name=blackmail_100
#SBATCH --partition=highram
#SBATCH --nodelist=node-12
#SBATCH --gpus=4
#SBATCH --mem=400G
#SBATCH --time=4:00:00
#SBATCH --output=/workspace-vast/annas/logs/blackmail_100pct_%j.out
#SBATCH --error=/workspace-vast/annas/logs/blackmail_100pct_%j.err

echo "========================================"
echo "BLACKMAIL STEERING - 100% STRENGTH"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "========================================"
echo "Running UA vectors then TEXT vectors"
echo "6 conditions each = 12 total"
echo "100 samples per condition = 1200 total"
echo "========================================"

# Load secrets for Anthropic API
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate environment
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

# Allow pickle serialization for steering hooks
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools

# Run UA vectors first
echo ""
echo "========================================"
echo "PHASE 1: UA DISENTANGLED VECTORS"
echo "========================================"
python -m experiments.steering.experiments.blackmail_steering_100pct \
    --vector-type ua \
    --layer 50 \
    --num-samples 100 \
    --max-tokens 4000 \
    --tensor-parallel 4 \
    --gpu-memory 0.90

# Run TEXT vectors second
echo ""
echo "========================================"
echo "PHASE 2: TEXT MEAN DIFF VECTORS"
echo "========================================"
python -m experiments.steering.experiments.blackmail_steering_100pct \
    --vector-type text \
    --layer 50 \
    --num-samples 100 \
    --max-tokens 4000 \
    --tensor-parallel 4 \
    --gpu-memory 0.90

echo "========================================"
echo "End time: $(date)"
echo "========================================"
