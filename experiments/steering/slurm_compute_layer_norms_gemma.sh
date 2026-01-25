#!/bin/bash
#SBATCH --job-name=layer_norms_gemma
#SBATCH --output=/workspace-vast/annas/logs/layer_norms_gemma_%j.out
#SBATCH --error=/workspace-vast/annas/logs/layer_norms_gemma_%j.err
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=2:00:00

# Compute layer norms for Gemma 27B (62 layers)

set -e

source /workspace-vast/annas/.secrets/load_secrets.sh

echo "=== Computing Layer Norms: Gemma 27B ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
date

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python -m experiments.steering.compute_all_layer_norms \
    --model gemma \
    --output experiments/steering/layer_norms.json

echo "Done!"
date
