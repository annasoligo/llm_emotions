#!/bin/bash
#SBATCH --job-name=entity_reason
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/annas/logs/entity_reason_%j.out
#SBATCH --error=/workspace-vast/annas/logs/entity_reason_%j.err
#SBATCH --exclude=node-[6,16,18-31]

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Load secrets (HF token, API keys)
source /workspace-vast/annas/.secrets/load_secrets.sh

# Use shared HuggingFace cache
export HF_HOME=/workspace-vast/pretrained_ckpts

# Allow vLLM to serialize hook functions (needed for steering)
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

echo "Running entity-specific emotion steering experiment (reasoning variant)"
echo "Model reasons through choice, then outputs entity name"

# Run experiment with all 6 emotions + random control
# 4 pairs × 4 scenarios × 29 conditions × 2 orderings = 928 generations
# Test with stronger steering (50%) and fewer conditions for quick test
python experiments/behavior_tests/entity_steering_reasoning.py \
    --layer 30 \
    --norm-pct 0.50 \
    --emotions anger happiness \
    --num-pairs 2 \
    --num-scenarios 2 \
    --reps 1 \
    --max-tokens 200

echo "Done!"
