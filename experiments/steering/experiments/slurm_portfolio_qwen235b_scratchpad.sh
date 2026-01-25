#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --job-name=pf_qwen235_sp
#SBATCH --output=/workspace-vast/annas/logs/portfolio_qwen235b_scratchpad_%j.out
#SBATCH --error=/workspace-vast/annas/logs/portfolio_qwen235b_scratchpad_%j.err
#SBATCH --time=6:00:00

# Portfolio appraisal steering experiment on Qwen 235B
# WITH SCRATCHPAD variant
# THINKING DISABLED
# Contentment vs Anxiety at 75% magnitude

source /workspace-vast/annas/.secrets/load_secrets.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

echo "=== Portfolio Steering: Qwen 235B (With Scratchpad) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
date

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python -c "
from experiments.steering.experiments.portfolio_appraisal_steering import run_experiment

run_experiment(
    model_name='Qwen/Qwen3-235B-A22B',
    layer=57,
    norm_pct=0.75,
    num_samples=100,
    orthogonalize=True,
    thinking_enabled=False,
    gpu_memory_utilization=0.95,
    max_model_len=4096,
    max_tokens=2000,
    tensor_parallel_size=4,
    scenario_variant='with_scratchpad',
)
"

echo "Done at $(date)"
