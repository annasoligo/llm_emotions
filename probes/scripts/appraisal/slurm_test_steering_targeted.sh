#!/bin/bash
#SBATCH --partition=highram
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --job-name=steer_targeted
#SBATCH --output=/workspace-vast/annas/logs/%j_steer_targeted.out
#SBATCH --error=/workspace-vast/annas/logs/%j_steer_targeted.err
#SBATCH --time=1:00:00

set -e

echo "=========================================="
echo "TARGETED APPRAISAL STEERING TEST"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "Started: $(date)"
echo ""

export HF_HOME=/workspace-vast/pretrained_ckpts
export CUDA_VISIBLE_DEVICES=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export PYTHONUNBUFFERED=1

python -u -m probes.scripts.appraisal.test_steering_targeted \
    --layer 30 \
    --norm-pct 0.15 \
    --n-samples 5 \
    --max-tokens 250 \
    --orthogonalize

echo ""
echo "=========================================="
echo "COMPLETED: $(date)"
echo "=========================================="
