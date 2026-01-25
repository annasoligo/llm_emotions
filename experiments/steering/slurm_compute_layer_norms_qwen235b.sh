#!/bin/bash
#SBATCH --job-name=layer_norms_qwen235b
#SBATCH --output=/workspace-vast/annas/logs/layer_norms_qwen235b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/layer_norms_qwen235b_%j.err
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --time=6:00:00

# Compute layer norms for Qwen 235B (94 layers)

set -e

source /workspace-vast/annas/.secrets/load_secrets.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

echo "=== Computing Layer Norms: Qwen 235B ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
date

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python -m experiments.steering.compute_all_layer_norms \
    --model qwen235b \
    --output experiments/steering/layer_norms.json

echo "Done!"
date
