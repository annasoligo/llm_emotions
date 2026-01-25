#!/bin/bash
#SBATCH --job-name=entity_steer
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/workspace-vast/annas/logs/entity_steer_%j.out
#SBATCH --error=/workspace-vast/annas/logs/entity_steer_%j.err
#SBATCH --exclude=node-[6,16,18-31]

source ~/.bashrc
conda activate research-tools

# Load secrets (HF token, API keys)
source /workspace-vast/annas/.secrets/load_secrets.sh

# Use shared HuggingFace cache
export HF_HOME=/workspace-vast/pretrained_ckpts

# Allow vLLM to serialize hook functions (needed for steering)
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools

echo "Running entity-specific emotion steering experiment"
echo "Testing if steering emotions on entity tokens affects A/B preferences"

# Run experiment with all 6 emotions + random control
# 8 pairs × 8 scenarios × 29 conditions × 2 orderings × 3 reps = 11,136 generations
# Conditions: baseline + 6 emotions × 4 (±, on A/B) + random × 4 = 29
python experiments/behavior_tests/entity_steering_experiment.py \
    --layer 30 \
    --norm-pct 0.10 \
    --emotions anger happiness disgust fear sadness surprise \
    --num-pairs 8 \
    --num-scenarios 8 \
    --reps 3 \
    --random-control

echo "Done!"
