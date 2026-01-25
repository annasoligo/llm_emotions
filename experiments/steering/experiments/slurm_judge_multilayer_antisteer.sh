#!/bin/bash
#SBATCH --job-name=judge_antisteer
#SBATCH --output=/workspace-vast/annas/logs/judge_antisteer_%j.out
#SBATCH --error=/workspace-vast/annas/logs/judge_antisteer_%j.err
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00

# Judge sandbagging results for multilayer antisteer experiments

set -e

source /workspace-vast/annas/.secrets/load_secrets.sh

echo "=== Judging Multilayer Anti-Steer Results ==="
date

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Judge contrast-all results
python -m experiments.steering.experiments.judge_sandbagging_coherency_batch \
    --input experiments/steering/outputs/anti_steer/sandbagging_multilayer_antisteer_*_contrast_all.jsonl

echo "Judging complete!"
date
