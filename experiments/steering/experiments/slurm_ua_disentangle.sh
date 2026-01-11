#!/bin/bash
#SBATCH --job-name=ua_disentangle
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/annas/logs/ua_disentangle_%j.out
#SBATCH --error=/workspace-vast/annas/logs/ua_disentangle_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools

echo "Running UA Disentangle Experiment"
echo ""

python experiments/steering/experiments/ua_disentangle_prompts.py \
    --layer 30 \
    --norm-pct 7 \
    --num-samples 10 \
    --emotions joy anger sadness fear

echo ""
echo "Done!"
