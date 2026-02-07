#!/bin/bash
#SBATCH --job-name=role_g27_base_named
#SBATCH --output=experiments/role_attribution/logs/gemma27b_base_named_%j.out
#SBATCH --error=experiments/role_attribution/logs/gemma27b_base_named_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python experiments/role_attribution/run_generation.py \
    --model-family gemma27b \
    --model-type base \
    --data experiments/role_attribution/outputs/prepared_statements_20260205_111142.json \
    --role named
