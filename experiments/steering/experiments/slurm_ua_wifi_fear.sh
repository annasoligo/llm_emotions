#!/bin/bash
#SBATCH --job-name=wifi_fear
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=0:30:00
#SBATCH --output=/workspace-vast/annas/logs/ua_wifi_fear_%j.out
#SBATCH --error=/workspace-vast/annas/logs/ua_wifi_fear_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools

python experiments/steering/experiments/ua_wifi_fear_test.py
