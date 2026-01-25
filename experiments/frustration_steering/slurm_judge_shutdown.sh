#!/bin/bash
#SBATCH --job-name=judge_shutdown
#SBATCH --output=/workspace-vast/annas/git/research-tools/experiments/frustration_steering/logs/judge_shutdown_%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/experiments/frustration_steering/logs/judge_shutdown_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

echo "========================================"
echo "Judging shutdown experiment responses"
echo "========================================"

# Load secrets
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate venv
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Activated venv"
fi

cd /workspace-vast/annas/git/research-tools/experiments/frustration_steering

# Judge all shutdown responses
echo "Judging all shutdown responses..."
python judge_responses.py \
    outputs/responses_shutdown_*.jsonl \
    --output outputs/judged_shutdown.jsonl

if [ $? -ne 0 ]; then
    echo "ERROR: Judging failed"
    exit 1
fi

echo "✓ Judging complete"
