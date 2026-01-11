#!/bin/bash
#SBATCH --job-name=coherency_L30
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=1:00:00
#SBATCH --output=/workspace-vast/annas/logs/coherency_L30_high_%j.out
#SBATCH --error=/workspace-vast/annas/logs/coherency_L30_high_%j.err

# Load secrets and environment
source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

# Required for vLLM steering hooks
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools

echo "Testing L30 at high percentages (ignoring formatting)"
echo ""

python experiments/steering/experiments/coherency_limit_sweep.py \
    --layer 30 \
    --num-samples 10 \
    --percentages 12 15 18 20 25

echo ""
echo "Done!"
