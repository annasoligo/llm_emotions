#!/bin/bash
#SBATCH --job-name=qwen32b_baseline
#SBATCH --output=/workspace-vast/annas/git/research-tools/experiments/steering/experiments/slurm_jobs/qwen32b_baseline_%j.out
#SBATCH --error=/workspace-vast/annas/git/research-tools/experiments/steering/experiments/slurm_jobs/qwen32b_baseline_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1:00:00

set -e

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "Running Qwen3-32B BASELINE (no steering)"
echo "========================================="
echo "Samples: 100"
echo ""

python -m experiments.steering.experiments.blackmail_qwen32b_baseline \
    --num-samples 100 \
    --tensor-parallel 2 \
    --max-model-len 16384

echo ""
echo "Baseline experiment complete!"
