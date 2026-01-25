#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --job-name=appr_dim_qwen32b
#SBATCH --output=/workspace-vast/annas/logs/%j_appraisal_dimension_qwen32b.out
#SBATCH --error=/workspace-vast/annas/logs/%j_appraisal_dimension_qwen32b.err
#SBATCH --time=2:00:00

set -e

echo "=========================================="
echo "APPRAISAL DIMENSION STEERING: Qwen3-32B"
echo "Testing 100%, 125%, 150% magnitudes"
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

OUTPUT_DIR=/workspace-vast/annas/appraisal_data/qwen32b

echo "=========================================="
echo "Running appraisal dimension steering..."
echo "=========================================="

python -u -m experiments.behavior_tests.appraisal_dimension_steering \
    --model Qwen/Qwen3-32B \
    --layer 30 \
    --axes valence uncertainty agency \
    --norm-pcts 1.0 1.25 1.50 \
    --num-seeds 10 \
    --activations $OUTPUT_DIR/activations.h5 \
    --metadata $OUTPUT_DIR/activation_metadata.json \
    --output-dir experiments/steering/outputs/appraisal_dimension

echo ""
echo "=========================================="
echo "COMPLETED: $(date)"
echo "=========================================="
