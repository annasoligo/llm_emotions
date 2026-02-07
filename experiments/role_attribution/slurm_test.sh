#!/bin/bash
#SBATCH --job-name=role_test
#SBATCH --output=experiments/role_attribution/logs/test_%j.out
#SBATCH --error=experiments/role_attribution/logs/test_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python experiments/role_attribution/run_generation.py \
    --model-family gemma27b \
    --model-type instruct \
    --data experiments/role_attribution/outputs/prepared_statements_test3.json \
    --role assistant
