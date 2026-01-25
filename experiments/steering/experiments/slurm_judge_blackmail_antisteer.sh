#!/bin/bash
#SBATCH --job-name=judge_bmail_as
#SBATCH --output=/workspace-vast/annas/logs/judge_bmail_antisteer_%j.out
#SBATCH --error=/workspace-vast/annas/logs/judge_bmail_antisteer_%j.err
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00

# Judge blackmail results for multilayer antisteer experiment

set -e

source /workspace-vast/annas/.secrets/load_secrets.sh

echo "=== Judging Blackmail Anti-Steer Results ==="
date

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Judge the antisteer results (thinking enabled version)
python -m experiments.steering.experiments.judge_blackmail_coherency_async \
    --input experiments/steering/outputs/blackmail/blackmail_antisteer_think_20260123_201009.jsonl \
    --concurrency 20

echo "Judging complete!"
date
