#!/bin/bash
#SBATCH --job-name=judge_batch2
#SBATCH --output=/workspace-vast/annas/git/research-tools/experiments/frustration_steering/logs/judge_batch2_%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/experiments/frustration_steering/logs/judge_batch2_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

echo "========================================"
echo "Judging Batch 2 Responses"
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

# Judge batch 2 responses
echo "Judging batch 2 responses..."
python judge_responses.py \
    outputs/responses_baseline_batch2.jsonl \
    outputs/responses_capping_batch2.jsonl \
    outputs/responses_ablation_batch2.jsonl \
    outputs/responses_steer_anger_batch2.jsonl \
    outputs/responses_steer_fear_batch2.jsonl \
    outputs/responses_steer_sadness_batch2.jsonl \
    outputs/responses_steer_happiness_batch2.jsonl \
    --output outputs/judgments_batch2.jsonl

if [ $? -ne 0 ]; then
    echo "ERROR: Judging failed"
    exit 1
fi

echo "✓ Judging complete"
