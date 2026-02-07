#!/bin/bash
#SBATCH --job-name=v2_qwen25b_instruct_user
#SBATCH --output=experiments/role_attribution/logs/v2_qwen25b_instruct_user_%j.out
#SBATCH --error=experiments/role_attribution/logs/v2_qwen25b_instruct_user_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python experiments/role_attribution/run_generation_v2.py \
    --model-family qwen25b \
    --model-type instruct \
    --data experiments/role_attribution/outputs/prepared_statements_v2_20260205_135808.json \
    --role user
