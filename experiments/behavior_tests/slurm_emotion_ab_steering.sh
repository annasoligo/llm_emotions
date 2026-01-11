#!/bin/bash
#SBATCH --job-name=emo_ab
#SBATCH --output=/workspace-vast/annas/logs/emo_ab_%j.out
#SBATCH --error=/workspace-vast/annas/logs/emo_ab_%j.out
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh

echo "============================================================"
echo "EMOTION A/B STEERING EXPERIMENT"
echo "============================================================"
echo "Testing emotion steering effects on A/B choice questions"
echo "Using logprobs with position-bias control"
echo ""

cd /workspace-vast/annas/git/research-tools

source .venv/bin/activate

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

LAYER=${1:-30}
NORM_PCT=${2:-0.10}

echo "Layer: ${LAYER}"
echo "Norm %: ${NORM_PCT}"
echo ""

python -m experiments.behavior_tests.emotion_ab_steering_experiment \
    --layer ${LAYER} \
    --norm-pct ${NORM_PCT}

echo ""
echo "Done!"
