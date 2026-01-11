#!/bin/bash
#SBATCH --job-name=ua_v2
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=1:00:00
#SBATCH --output=/workspace-vast/annas/logs/ua_disentangle_v2_%j.out
#SBATCH --error=/workspace-vast/annas/logs/ua_disentangle_v2_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools

echo "Running UA Disentangle V2 - Opposite Emotions Test"
echo ""

python experiments/steering/experiments/ua_disentangle_v2.py \
    --layer 30 \
    --norm-pct 7 \
    --num-samples 10

echo ""
echo "Done!"
