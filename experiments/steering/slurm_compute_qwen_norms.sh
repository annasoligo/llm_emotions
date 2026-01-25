#!/bin/bash
#SBATCH --job-name=qwen_norms
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=1:00:00
#SBATCH --output=/workspace-vast/annas/logs/qwen_norms_%j.out
#SBATCH --error=/workspace-vast/annas/logs/qwen_norms_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools

echo "Computing Qwen layer norms..."

python -m experiments.steering.compute_qwen_layer_norms --layers 30

echo "Done!"
