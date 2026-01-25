#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --job-name=pf_qwen235b_nothink
#SBATCH --output=/workspace-vast/annas/logs/portfolio_qwen235b_nothink_%j.out
#SBATCH --error=/workspace-vast/annas/logs/portfolio_qwen235b_nothink_%j.err
#SBATCH --time=8:00:00

# Portfolio appraisal steering experiment on Qwen 235B
# THINKING DISABLED
# Contentment vs Anxiety at 75% magnitude

source /workspace-vast/annas/.secrets/load_secrets.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_V1=0

echo "=== Portfolio Steering: Qwen 235B (No Think) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
date

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Run experiment: 75% steering, 100 samples per condition
python -c "
from experiments.steering.experiments.portfolio_appraisal_steering import run_experiment

run_experiment(
    model_name='Qwen/Qwen3-235B-A22B',
    layer=50,
    norm_pct=0.75,
    num_samples=100,
    orthogonalize=True,
    thinking_enabled=False,
    gpu_memory_utilization=0.90,
    max_model_len=8192,
    max_tokens=2000,
    tensor_parallel_size=4,
    scenario_variant='with_scratchpad',
)
"

echo "Done at $(date)"
