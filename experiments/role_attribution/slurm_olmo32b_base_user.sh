#!/bin/bash
#SBATCH --job-name=role_olm_base_user
#SBATCH --output=experiments/role_attribution/logs/olmo32b_base_user_%j.out
#SBATCH --error=experiments/role_attribution/logs/olmo32b_base_user_%j.err
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
    --model-family olmo32b \
    --model-type base \
    --data experiments/role_attribution/outputs/prepared_statements_20260205_111142.json \
    --role user
