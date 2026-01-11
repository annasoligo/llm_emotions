#!/bin/bash
#SBATCH --job-name=sb_random
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=/workspace-vast/annas/logs/sb_random_%j.out
#SBATCH --error=/workspace-vast/annas/logs/sb_random_%j.err

# Load secrets and environment
source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

# Required for vLLM steering hooks
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd /workspace-vast/annas/git/research-tools

echo "============================================================"
echo "SANDBAGGING RANDOM VECTORS EXPERIMENT"
echo "============================================================"
echo "Testing all 3 formats with random vector steering"
echo ""

# Run layer 30 only (main comparison layer)
echo "Running layer 30..."
python -m experiments.steering.experiments.sandbagging_random_vectors_experiment \
    --layer 30 \
    --norm-pct 0.10 \
    --num-samples 5 \
    --num-random-vectors 3

echo ""
echo "Done!"
