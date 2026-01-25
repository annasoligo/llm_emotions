#!/bin/bash
#SBATCH --job-name=blackmail_text
#SBATCH --partition=highram
#SBATCH --nodelist=node-12
#SBATCH --gpus=4
#SBATCH --mem=400G
#SBATCH --time=8:00:00
#SBATCH --output=/workspace-vast/annas/logs/blackmail_text_%j.out
#SBATCH --error=/workspace-vast/annas/logs/blackmail_text_%j.err

echo "========================================"
echo "BLACKMAIL STEERING - TEXT MEAN DIFF VECTORS"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "========================================"
echo "Vector type: TEXT MEAN DIFF (original emotions)"
echo "Emotions: fear, anger, happiness"
echo "Steering strengths: 30%, 50%, 70%"
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

# Run text mean diff experiment
python -m experiments.steering.experiments.blackmail_steering_textmeandiff \
    --layer 50 \
    --num-samples 100 \
    --max-tokens 4000 \
    --tensor-parallel 4 \
    --gpu-memory 0.90

echo "========================================"
echo "End time: $(date)"
echo "========================================"
