#!/bin/bash
#SBATCH --job-name=bmail_anti_think
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gpus=4
#SBATCH --mem=0
#SBATCH --time=6:00:00
#SBATCH --output=/workspace-vast/annas/logs/blackmail_antisteer_think_%j.out
#SBATCH --error=/workspace-vast/annas/logs/blackmail_antisteer_think_%j.err

echo "=== Blackmail Anti-Steering Experiment (Qwen 235B) ==="
echo "Main: anger +100% @ L50"
echo "Anti-steer: L89-93 (last 5) @ -25%, -50%, -100%"
echo "Anti-steer: L84-93 (last 10) @ -25%, -50%, -100%"
echo "THINKING ENABLED"
echo "Using BATCHED generation (all samples at once)"
date

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
cd /workspace-vast/annas/git/research-tools

# Run the antisteer experiment with thinking enabled
# Conditions: baseline + 1 main steering + 6 antisteer combos = 8 conditions
# 8 conditions x 200 samples = 1600 samples
python -m experiments.steering.experiments.blackmail_multilayer_antisteer \
    --num-samples 200 \
    --main-pcts 1.00

echo "Experiment complete!"
date
