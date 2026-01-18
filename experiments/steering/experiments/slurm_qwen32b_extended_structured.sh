#!/bin/bash
#SBATCH --job-name=qwen32b_extended_struct
#SBATCH --output=/workspace-vast/annas/git/research-tools/experiments/steering/experiments/slurm_jobs/qwen32b_extended_struct_%j.out
#SBATCH --error=/workspace-vast/annas/git/research-tools/experiments/steering/experiments/slurm_jobs/qwen32b_extended_struct_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=4:00:00

set -e

# Required for vLLM v1 steering hook serialization
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "Running Qwen3-32B EXTENDED blackmail steering - STRUCTURED"
echo "==========================================================="
echo "Conditions: emotion at 100%, 125%, 150%, 200%"
echo "           random at 100%, 200%"
echo "Samples: 100 per condition"
echo ""

python -m experiments.steering.experiments.blackmail_steering_qwen32b_extended \
    --layer 30 \
    --num-samples 100 \
    --tensor-parallel 2 \
    --max-model-len 16384 \
    --structured

echo ""
echo "Extended structured experiment complete!"
