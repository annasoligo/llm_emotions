#!/bin/bash
#SBATCH --job-name=fear_cap
#SBATCH --output=/workspace-vast/annas/logs/fear_capping_%j.out
#SBATCH --error=/workspace-vast/annas/logs/fear_capping_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2:00:00

# Fear capping experiment
# Instead of anti-steering (subtracting fear), we cap activations along fear direction

set -e

source /workspace-vast/annas/.secrets/load_secrets.sh
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

echo "=== Fear Capping Experiment ==="
echo "Fear steering: L30 @ +7.5%"
echo "Capping at L57-61 with thresholds: 0.2, 0.5, 1.0 (in std units)"
date

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

python -m experiments.steering.experiments.sandbagging_fear_capping \
    --main-layer 30 \
    --main-pct 0.075 \
    --capping-layers 57 58 59 60 61 \
    --cap-thresholds 0.2 0.5 1.0 \
    --num-samples 20

echo "Experiment complete!"
date
