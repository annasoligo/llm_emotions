#!/bin/bash
#SBATCH --job-name=antisteer_L2
#SBATCH --output=/workspace-vast/annas/logs/antisteer_last2_%j.out
#SBATCH --error=/workspace-vast/annas/logs/antisteer_last2_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2:00:00

# Anti-steer ONLY at last 2 layers (L60-61) with higher magnitudes (-10%, -15%)
# Hypothesis: Expression is shaped at the very last layers
# Goal: Suppress fear expression while retaining behavioral change (accuracy drop)

set -e

source /workspace-vast/annas/.secrets/load_secrets.sh

# Allow pickle-based serialization for vLLM apply_model
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

echo "=== Anti-Steer Last 2 Layers Experiment ==="
echo "Fear steering: L30 @ +7.5%"
echo "Anti-steering: L60-61 @ -10%, -15%"
date

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

python -m experiments.steering.experiments.sandbagging_multilayer_antisteer \
    --vector-type textmeandiff \
    --main-layer 30 \
    --main-pct 0.075 \
    --antisteer-layers 60 61 \
    --antisteer-pcts 0.10 0.15 \
    --num-samples 20

echo "Experiment complete!"
date
