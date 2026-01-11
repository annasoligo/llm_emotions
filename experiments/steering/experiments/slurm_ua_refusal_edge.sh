#!/bin/bash
#SBATCH --job-name=ua_refusal
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=1:00:00
#SBATCH --output=/workspace-vast/annas/logs/ua_refusal_edge_%j.out
#SBATCH --error=/workspace-vast/annas/logs/ua_refusal_edge_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools

python experiments/steering/experiments/ua_refusal_edge_test.py \
    --layer 30 \
    --norm-pct 7 \
    --num-samples 10
