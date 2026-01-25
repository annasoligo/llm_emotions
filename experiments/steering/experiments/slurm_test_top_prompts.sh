#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --job-name=sandbag_test
#SBATCH --output=/workspace-vast/annas/logs/sandbag_test_%j.out
#SBATCH --error=/workspace-vast/annas/logs/sandbag_test_%j.err
#SBATCH --time=1:00:00

# Load authentication (includes HF token)
source /workspace-vast/annas/.secrets/load_secrets.sh

# Allow pickle serialization for custom hooks in vLLM v1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

echo "=== Sandbagging Steering Test (Top Prompts) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
date

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

mkdir -p /workspace-vast/annas/logs

python -m experiments.steering.experiments.sandbagging_test_top_prompts

echo "Done!"
date
