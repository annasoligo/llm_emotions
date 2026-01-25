#!/bin/bash
#SBATCH --job-name=appr_bin_gemma
#SBATCH --output=/workspace-vast/annas/logs/%j_appraisal_binary_gemma27b.out
#SBATCH --error=/workspace-vast/annas/logs/%j_appraisal_binary_gemma27b.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gpus=1
#SBATCH --time=2:00:00

echo "=========================================="
echo "BINARY APPRAISAL STEERING: Gemma-3-27B"
echo "Testing 5%, 7%, 10% magnitudes"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Started: $(date)"

cd /workspace-vast/annas/git/research-tools

source .venv/bin/activate

export HF_HOME=/workspace-vast/pretrained_ckpts
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_V1=0

echo ""
echo "=========================================="
echo "Running binary appraisal steering..."
echo "=========================================="

python -m experiments.behavior_tests.appraisal_dimension_steering_binary \
    --model google/gemma-3-27b-it \
    --norm-pcts 0.05 0.07 0.10 \
    --num-seeds 4

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
