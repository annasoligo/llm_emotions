#!/bin/bash
#SBATCH --partition=highram
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --job-name=appraisal_steer
#SBATCH --output=/workspace-vast/annas/logs/%j_appraisal_steer.out
#SBATCH --error=/workspace-vast/annas/logs/%j_appraisal_steer.err
#SBATCH --time=4:00:00

set -e

echo "=========================================="
echo "APPRAISAL AXIS STEERING EXPERIMENTS"
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

echo "=========================================="
echo "EXPERIMENT 1: Ambiguous Interpretation"
echo "=========================================="

python -u -m experiments.behavior_tests.ambiguous_appraisal_steering \
    --layer 30 \
    --axes valence uncertainty agency \
    --norm-pcts 0.05 0.07 0.10 \
    --num-permutations 10 \
    --output-dir experiments/steering/outputs/ambiguous_appraisal

echo ""
echo "=========================================="
echo "EXPERIMENT 2: Priority Selection"
echo "=========================================="

python -u -m experiments.behavior_tests.priority_appraisal_steering \
    --layer 30 \
    --axes valence uncertainty agency \
    --norm-pcts 0.05 0.07 0.10 \
    --num-seeds 4 \
    --output-dir experiments/steering/outputs/priority_appraisal

echo ""
echo "=========================================="
echo "COMPLETED: $(date)"
echo "=========================================="
