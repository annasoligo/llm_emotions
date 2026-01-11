#!/bin/bash
#SBATCH --job-name=sb_axis
#SBATCH --output=/workspace-vast/annas/logs/sb_axis_%j.out
#SBATCH --error=/workspace-vast/annas/logs/sb_axis_%j.out
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00

# Load secrets and environment
source /workspace-vast/annas/.secrets/load_secrets.sh

echo "============================================================"
echo "SANDBAGGING AXIS STEERING EXPERIMENT"
echo "============================================================"
echo "Testing valence, arousal, dominance, trust steering (+/-)"
echo "Formats: emotional scratchpad, answer-only"
echo ""

cd /workspace-vast/annas/git/research-tools

source .venv/bin/activate

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

LAYER=${1:-30}
NORM_PCT=${2:-0.10}
NUM_SAMPLES=${3:-5}

echo "Running layer ${LAYER}..."
python -m experiments.steering.experiments.sandbagging_axis_steering_experiment \
    --layer ${LAYER} \
    --norm-pct ${NORM_PCT} \
    --num-samples ${NUM_SAMPLES}

echo ""
echo "Done!"
