#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --job-name=pf_qwen32_1gpu
#SBATCH --output=/workspace-vast/annas/logs/portfolio_qwen32b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/portfolio_qwen32b_%j.err
#SBATCH --time=6:00:00

# Portfolio appraisal steering experiment on Qwen 32B
# Single GPU version to avoid NCCL issues
# Contentment vs Anxiety at 75% magnitude
# Uses Qwen-specific appraisal vectors

source /workspace-vast/annas/.secrets/load_secrets.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

echo "=== Portfolio Steering: Qwen 32B (Single GPU) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
date

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Run experiment: 75% steering, 100 samples per condition
# Single GPU (tensor_parallel_size=1) to avoid NCCL issues
# Model-specific appraisal vectors are auto-detected
python -c "
import os
os.environ['VLLM_USE_V1'] = '0'
from experiments.steering.experiments.portfolio_appraisal_steering import run_experiment

run_experiment(
    model_name='Qwen/Qwen3-32B',
    layer=30,
    norm_pct=0.75,
    num_samples=100,
    # Paths auto-detected from APPRAISAL_DATA_PATHS for Qwen
    orthogonalize=True,
    thinking_enabled=True,
    gpu_memory_utilization=0.95,
    max_model_len=4096,
    max_tokens=2000,
    tensor_parallel_size=1,
    scenario_variant='with_scratchpad',
)
"

echo "Done at $(date)"
