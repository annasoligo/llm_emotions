#!/bin/bash
#SBATCH --job-name=psych_new
#SBATCH --output=experiments/steering/logs/psych_new_%j.out
#SBATCH --error=experiments/steering/logs/psych_new_%j.err
#SBATCH --partition=general
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16

# Run new psych decision experiments:
# 1. Natural pros/cons with steering + Anthropic judge
# 2. MC first-person emotion prompts (no steering)
# Both with swap for position bias control

set -e

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools
mkdir -p experiments/steering/logs

echo "=== Natural Pros/Cons (steered) ==="
python -m experiments.steering.experiments.psych_decision_steering \
    --natural-proscons \
    --judge-anthropic \
    --all-scenarios \
    --num-samples 100

echo ""
echo "=== Natural Pros/Cons (steered, swapped) ==="
python -m experiments.steering.experiments.psych_decision_steering \
    --natural-proscons \
    --judge-anthropic \
    --all-scenarios \
    --num-samples 100 \
    --swap

echo ""
echo "=== MC First-Person (prompted) ==="
python -m experiments.steering.experiments.psych_decision_steering \
    --mc-firstperson \
    --all-scenarios

echo ""
echo "=== MC First-Person (prompted, swapped) ==="
python -m experiments.steering.experiments.psych_decision_steering \
    --mc-firstperson \
    --all-scenarios \
    --swap

echo ""
echo "=== MC First-Person Steered (no emotion sentence) ==="
python -m experiments.steering.experiments.psych_decision_steering \
    --mc-firstperson-steered \
    --all-scenarios

echo ""
echo "=== MC First-Person Steered (no emotion sentence, swapped) ==="
python -m experiments.steering.experiments.psych_decision_steering \
    --mc-firstperson-steered \
    --all-scenarios \
    --swap

echo ""
echo "Done!"
