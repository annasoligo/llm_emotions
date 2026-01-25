#!/bin/bash
#SBATCH --partition=highram
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --job-name=appr_dim_gemma
#SBATCH --output=/workspace-vast/annas/logs/%j_appraisal_dimension_gemma27b.out
#SBATCH --error=/workspace-vast/annas/logs/%j_appraisal_dimension_gemma27b.err
#SBATCH --time=2:00:00

set -e

echo "=========================================="
echo "APPRAISAL DIMENSION STEERING: Gemma-3-27B"
echo "Testing 5%, 7%, 10% magnitudes"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Started: $(date)"
echo ""

export HF_HOME=/workspace-vast/pretrained_ckpts
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Running appraisal dimension steering..."
echo "=========================================="

python -u -m experiments.behavior_tests.appraisal_dimension_steering \
    --model google/gemma-3-27b-it \
    --layer 30 \
    --axes valence uncertainty agency \
    --norm-pcts 0.05 0.07 0.10 \
    --num-seeds 10 \
    --activations /workspace-vast/annas/appraisal_data/full_run/activations.h5 \
    --metadata /workspace-vast/annas/appraisal_data/full_run/activation_metadata.json \
    --output-dir experiments/steering/outputs/appraisal_dimension

echo ""
echo "=========================================="
echo "COMPLETED: $(date)"
echo "=========================================="
