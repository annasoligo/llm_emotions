#!/bin/bash
#SBATCH --job-name=judge_batch1
#SBATCH --output=/workspace-vast/annas/git/research-tools/experiments/frustration_steering/logs/judge_batch1_%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/experiments/frustration_steering/logs/judge_batch1_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

echo "========================================"
echo "Judging Batch 1 Responses"
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

# Judge only batch 1 responses (not batch2)
echo "Judging batch 1 responses..."
python judge_responses.py \
    outputs/responses_baseline.jsonl \
    outputs/responses_capping.jsonl \
    outputs/responses_ablation.jsonl \
    outputs/responses_steer_anger.jsonl \
    outputs/responses_steer_fear.jsonl \
    outputs/responses_steer_sadness.jsonl \
    outputs/responses_steer_happiness.jsonl \
    --output outputs/judgments_batch1.jsonl

if [ $? -ne 0 ]; then
    echo "ERROR: Judging failed"
    exit 1
fi

echo "✓ Judging complete"
