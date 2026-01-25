#!/bin/bash
#SBATCH --job-name=bmail_anti
#SBATCH --output=/workspace-vast/annas/logs/blackmail_antisteer_%j.out
#SBATCH --error=/workspace-vast/annas/logs/blackmail_antisteer_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=4:00:00

# Blackmail anti-steering experiment on Qwen 235B
# Main steering: anger +50% and +100% at layer 50
# Anti-steering: layers 88-93 (last 6 layers) with -25%, -50%

set -e

source /workspace-vast/annas/.secrets/load_secrets.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

echo "=== Blackmail Anti-Steering Experiment (Qwen 235B) ==="
echo "Main: anger +50% and +100% @ L50"
echo "Anti-steer: L88-93 @ -25%, -50%"
date

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

python -m experiments.steering.experiments.blackmail_multilayer_antisteer \
    --num-samples 50 \
    --main-pcts 0.50 1.00 \
    --antisteer-pcts -0.25 -0.50 \
    --antisteer-layers 88 89 90 91 92 93

echo "Experiment complete!"
date
