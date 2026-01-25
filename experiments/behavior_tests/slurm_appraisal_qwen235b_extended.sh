#!/bin/bash
#SBATCH --partition=highram
#SBATCH --qos=high
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --job-name=appraisal_235b_ext
#SBATCH --output=/workspace-vast/annas/logs/%j_appraisal_qwen235b_extended.out
#SBATCH --error=/workspace-vast/annas/logs/%j_appraisal_qwen235b_extended.err
#SBATCH --time=8:00:00

set -e

echo "=========================================="
echo "APPRAISAL STEERING EXTENDED: Qwen3-235B"
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
echo "Running extended steering experiments..."
echo "=========================================="

# Ambiguous Interpretation with 100%, 125%, 150% (includes combo conditions)
python -u -m experiments.behavior_tests.ambiguous_appraisal_steering \
    --layer 50 \
    --axes valence uncertainty agency \
    --norm-pcts 1.0 1.25 1.50 \
    --num-permutations 10 \
    --activations $OUTPUT_DIR/activations.h5 \
    --metadata $OUTPUT_DIR/activation_metadata.json \
    --output-dir experiments/steering/outputs/ambiguous_appraisal \
    --model Qwen/Qwen3-235B-A22B

echo ""
echo "=========================================="
echo "COMPLETED: $(date)"
echo "=========================================="
