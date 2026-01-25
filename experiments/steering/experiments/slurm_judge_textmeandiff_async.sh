#!/bin/bash
#SBATCH --job-name=judge_tmd_async
#SBATCH --output=/workspace-vast/annas/logs/judge_tmd_async_%j.out
#SBATCH --error=/workspace-vast/annas/logs/judge_tmd_async_%j.err
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00

# Judge sandbagging results using async API with 50 concurrent requests

set -e

source /workspace-vast/annas/.secrets/load_secrets.sh

echo "=== Judging Textmeandiff Results (Async, 50 concurrent) ==="
date

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

INPUT_FILE="experiments/steering/outputs/anti_steer/sandbagging_multilayer_antisteer_20260122_203204_textmeandiff.jsonl"

echo "Input file: $INPUT_FILE"
echo "Samples: $(wc -l < $INPUT_FILE)"

python -m experiments.steering.experiments.judge_sandbagging_coherency_async \
    --input "$INPUT_FILE" \
    --concurrency 50

echo "Judging complete!"
date
