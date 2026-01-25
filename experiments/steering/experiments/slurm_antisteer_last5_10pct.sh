#!/bin/bash
#SBATCH --job-name=anti_L5_10
#SBATCH --output=/workspace-vast/annas/logs/antisteer_last5_10pct_%j.out
#SBATCH --error=/workspace-vast/annas/logs/antisteer_last5_10pct_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2:00:00

# Anti-steer at last 5 layers (L57-61) with -10%
# Testing stronger magnitude at more layers

set -e

source /workspace-vast/annas/.secrets/load_secrets.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

echo "=== Anti-Steer Last 5 Layers @ -10% ==="
echo "Fear steering: L30 @ +7.5%"
echo "Anti-steering: L57-61 @ -10%"
date

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

python -m experiments.steering.experiments.sandbagging_multilayer_antisteer \
    --vector-type textmeandiff \
    --main-layer 30 \
    --main-pct 0.075 \
    --antisteer-layers 57 58 59 60 61 \
    --antisteer-pcts 0.10 \
    --num-samples 20

echo "Experiment complete!"
date
