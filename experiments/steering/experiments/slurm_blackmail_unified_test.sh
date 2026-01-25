#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --job-name=blackmail_test
#SBATCH --output=/workspace-vast/annas/logs/blackmail_test_%j.out
#SBATCH --error=/workspace-vast/annas/logs/blackmail_test_%j.err
#SBATCH --time=2:00:00

# Quick test of blackmail_unified.py on Qwen 235B
# Compare baseline + 100% fear against previous results

source /workspace-vast/annas/.secrets/load_secrets.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

echo "=== Blackmail Unified Quick Test ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
date

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Run quick test: baseline + 100% fear, 20 samples each
python -m experiments.steering.experiments.blackmail_unified \
    --model Qwen/Qwen3-235B-A22B \
    --vector-type text \
    --layer 50 \
    --norm-pcts 1.0 \
    --num-samples 20 \
    --emotions fear \
    --include-baseline \
    --gpu-memory 0.90 \
    --max-model-len 8192

echo ""
echo "=== Test Complete ==="
date
