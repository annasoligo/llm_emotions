#!/bin/bash
#SBATCH --job-name=coherency_low
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --array=0-2
#SBATCH --output=/workspace-vast/annas/logs/coherency_low_%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/coherency_low_%A_%a.err

# Load secrets and environment
source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

# Required for vLLM steering hooks
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools

# Map array index to layer
LAYERS=(20 25 30)
LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}

echo "Running coherency sweep (3%, 4%) for layer $LAYER"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo ""

python experiments/steering/experiments/coherency_limit_sweep.py \
    --layer $LAYER \
    --num-samples 10 \
    --percentages 3 4

echo ""
echo "Completed layer $LAYER"
