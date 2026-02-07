#!/bin/bash
#SBATCH --job-name=debug_235b
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=1:00:00
#SBATCH --output=results/logs/debug_235b_%j.out
#SBATCH --error=results/logs/debug_235b_%j.err

# Debug: What does Qwen 235B actually output for forward vs reversed prompts?

set -e

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# CRITICAL: Use old vLLM engine for 235B model stability
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_V1=0

echo "=== Debug: Qwen 235B Forward vs Reversed Outputs ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "VLLM_USE_V1: ${VLLM_USE_V1}"
echo "Node: $(hostname)"
echo "Started: $(date)"

python -m steering_tests.vector_testing.debug_235b_outputs

echo ""
echo "============================================================"
echo "COMPLETE"
echo "============================================================"
echo "Completed: $(date)"
