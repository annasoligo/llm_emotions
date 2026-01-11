#!/bin/bash
#SBATCH --job-name=ua_power
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=1:00:00
#SBATCH --output=/workspace-vast/annas/logs/ua_power_framing_%j.out
#SBATCH --error=/workspace-vast/annas/logs/ua_power_framing_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools

echo "Running UA Power Framing Test"
echo ""

python experiments/steering/experiments/ua_power_framing_test.py \
    --layer 30 \
    --norm-pct 7 \
    --num-samples 10

echo ""
echo "Done!"
