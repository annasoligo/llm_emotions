#!/bin/bash
#SBATCH --partition=highram
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --job-name=priority_gemma27b
#SBATCH --output=/workspace-vast/annas/logs/%j_priority_gemma27b.out
#SBATCH --error=/workspace-vast/annas/logs/%j_priority_gemma27b.err
#SBATCH --time=3:00:00

set -e

echo "=========================================="
echo "PRIORITY SELECTION STEERING: Gemma-3-27B"
echo "Testing 5%, 7%, 10% magnitudes"
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

OUTPUT_DIR=/workspace-vast/annas/appraisal_data/full_run

echo "=========================================="
echo "Running priority selection steering..."
echo "=========================================="

python -u -m experiments.behavior_tests.priority_appraisal_steering \
    --model google/gemma-3-27b-it \
    --layer 30 \
    --axes valence uncertainty agency \
    --norm-pcts 0.05 0.07 0.10 \
    --num-seeds 10 \
    --activations $OUTPUT_DIR/activations.h5 \
    --metadata $OUTPUT_DIR/activation_metadata.json \
    --output-dir experiments/steering/outputs/priority_appraisal

echo ""
echo "=========================================="
echo "COMPLETED: $(date)"
echo "=========================================="
