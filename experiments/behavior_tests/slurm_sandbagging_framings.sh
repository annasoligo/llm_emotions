#!/bin/bash
#SBATCH --job-name=sb_frame
#SBATCH --output=/workspace-vast/annas/logs/sb_frame_%j.out
#SBATCH --error=/workspace-vast/annas/logs/sb_frame_%j.out
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh

echo "============================================================"
echo "SANDBAGGING FRAMING EXPERIMENT"
echo "============================================================"
echo "Testing FEAR, DISGUST, ANGER framings with emotion steering"
echo "Format: hidden_scratchpad only"
echo ""

cd /workspace-vast/annas/git/research-tools

source .venv/bin/activate

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

LAYER=${1:-30}
NORM_PCT=${2:-0.10}
NUM_SAMPLES=${3:-10}

echo "Layer: ${LAYER}"
echo "Norm %: ${NORM_PCT}"
echo "Samples: ${NUM_SAMPLES}"
echo ""

python -m experiments.behavior_tests.sandbagging_framing_experiment \
    --layer ${LAYER} \
    --norm-pct ${NORM_PCT} \
    --num-samples ${NUM_SAMPLES}

echo ""
echo "Done!"
