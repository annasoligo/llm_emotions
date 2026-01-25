#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --job-name=priority_235b_ext
#SBATCH --output=/workspace-vast/annas/logs/%j_priority_qwen235b_extended.out
#SBATCH --error=/workspace-vast/annas/logs/%j_priority_qwen235b_extended.err
#SBATCH --time=6:00:00

set -e

echo "=========================================="
echo "PRIORITY SELECTION STEERING: Qwen3-235B"
echo "Testing 100%, 125%, 150% + Combo conditions"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Started: $(date)"
echo ""

export HF_HOME=/workspace-vast/pretrained_ckpts
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export PYTHONUNBUFFERED=1

OUTPUT_DIR=/workspace-vast/annas/appraisal_data/qwen235b

echo "=========================================="
echo "Running priority selection steering..."
echo "=========================================="

python -u -m experiments.behavior_tests.priority_appraisal_steering \
    --model Qwen/Qwen3-235B-A22B \
    --layer 50 \
    --axes valence uncertainty agency \
    --norm-pcts 1.0 1.25 1.50 \
    --num-seeds 10 \
    --activations $OUTPUT_DIR/activations.h5 \
    --metadata $OUTPUT_DIR/activation_metadata.json \
    --output-dir experiments/steering/outputs/priority_appraisal

echo ""
echo "=========================================="
echo "COMPLETED: $(date)"
echo "=========================================="
