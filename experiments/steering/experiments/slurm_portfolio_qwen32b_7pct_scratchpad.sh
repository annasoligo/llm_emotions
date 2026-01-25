#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --job-name=pf_q32_7p_sp
#SBATCH --output=/workspace-vast/annas/logs/portfolio_qwen32b_7pct_scratchpad_%j.out
#SBATCH --error=/workspace-vast/annas/logs/portfolio_qwen32b_7pct_scratchpad_%j.err
#SBATCH --time=2:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_V1=0

echo "=== Portfolio Steering: Qwen 32B 7% (With Scratchpad) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
date

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python -c "
from experiments.steering.experiments.portfolio_appraisal_steering import run_experiment

run_experiment(
    model_name='Qwen/Qwen3-32B',
    layer=30,
    norm_pct=0.07,
    num_samples=100,
    orthogonalize=True,
    thinking_enabled=False,
    gpu_memory_utilization=0.95,
    max_model_len=4096,
    max_tokens=2000,
    scenario_variant='with_scratchpad',
)
"

echo "Done at $(date)"
