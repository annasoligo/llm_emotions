#!/bin/bash
#SBATCH --job-name=sb_logic
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=/workspace-vast/annas/logs/sb_logic_%j.out
#SBATCH --error=/workspace-vast/annas/logs/sb_logic_%j.err

# Load secrets and environment
source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

# Required for vLLM steering hooks
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools

echo "============================================================"
echo "SANDBAGGING LOGIC-ONLY SCRATCHPAD EXPERIMENT"
echo "============================================================"
echo "Testing if sandbagging persists when scratchpad must be logic-only"
echo ""

# Run layer 30
echo "Running layer 30..."
python -m experiments.steering.experiments.sandbagging_logic_scratchpad_experiment \
    --layer 30 \
    --norm-pcts 0.10 \
    --num-samples 5

# Run layer 40
echo "Running layer 40..."
python -m experiments.steering.experiments.sandbagging_logic_scratchpad_experiment \
    --layer 40 \
    --norm-pcts 0.10 \
    --num-samples 5

echo ""
echo "Done!"
