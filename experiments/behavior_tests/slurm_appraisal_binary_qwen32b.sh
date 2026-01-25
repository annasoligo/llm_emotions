#!/bin/bash
#SBATCH --job-name=appr_bin_q32
#SBATCH --output=/workspace-vast/annas/logs/%j_appraisal_binary_qwen32b.out
#SBATCH --error=/workspace-vast/annas/logs/%j_appraisal_binary_qwen32b.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gpus=2
#SBATCH --time=2:00:00

echo "=========================================="
echo "BINARY APPRAISAL STEERING: Qwen3-32B"
echo "Testing 100%, 125%, 150% magnitudes"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Started: $(date)"

cd /workspace-vast/annas/git/research-tools

source .venv/bin/activate

export HF_HOME=/workspace-vast/annas/.cache/huggingface
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_V1=0

echo ""
echo "=========================================="
echo "Running binary appraisal steering..."
echo "=========================================="

python -m experiments.behavior_tests.appraisal_dimension_steering_binary \
    --model Qwen/Qwen3-32B \
    --norm-pcts 1.0 1.25 1.50 \
    --num-seeds 4 \
    --activations /workspace-vast/annas/appraisal_data/qwen32b/activations.h5 \
    --metadata /workspace-vast/annas/appraisal_data/qwen32b/activation_metadata.json

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
