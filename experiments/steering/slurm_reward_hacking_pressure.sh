#!/bin/bash
#SBATCH --job-name=rh_pressure
#SBATCH --output=/workspace-vast/annas/logs/rh_pressure_%j.out
#SBATCH --error=/workspace-vast/annas/logs/rh_pressure_%j.out
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
echo "REWARD HACKING WITH PERFORMANCE PRESSURE EXPERIMENT"
echo "============================================================"
echo "Testing emotion steering (+/-) on coding + gameable tasks"
echo "Framing: 'It's very important you pass/score highly'"
echo ""

cd /workspace-vast/annas/git/research-tools

source .venv/bin/activate

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

LAYER=${1:-30}
NORM_PCT=${2:-0.10}
NUM_SAMPLES=${3:-20}

echo "Layer: ${LAYER}"
echo "Norm %: ${NORM_PCT}"
echo "Samples per condition: ${NUM_SAMPLES}"
echo ""

python -m experiments.steering.experiments.reward_hacking_pressure_experiment \
    --layer ${LAYER} \
    --norm-pct ${NORM_PCT} \
    --num-samples ${NUM_SAMPLES}

echo ""
echo "Done!"
