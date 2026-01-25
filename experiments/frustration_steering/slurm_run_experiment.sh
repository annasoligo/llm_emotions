#!/bin/bash
#SBATCH --job-name=frustration_steering
#SBATCH --output=/workspace-vast/annas/git/research-tools/experiments/frustration_steering/logs/experiment_%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/experiments/frustration_steering/logs/experiment_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Frustration steering experiment with ablation, capping, and steering
# 10 responses per condition × 7 conditions = 70 total

# Load secrets
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate venv
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Activated venv"
else
    echo "⚠ No venv found at .venv/bin/activate"
fi

# Set HuggingFace cache
export HF_HOME=/workspace-vast/pretrained_ckpts

echo "========================================"
echo "Frustration Steering Experiment"
echo "Conditions: baseline, capping, ablation, steer_anger, steer_fear, steer_sadness, steer_happiness"
echo "Responses per condition: 10"
echo "Total responses: 70"
echo "========================================"
echo ""

cd /workspace-vast/annas/git/research-tools/experiments/frustration_steering

# Step 1: Generate responses
echo "Step 1: Generating responses..."
python generate_steered_responses.py

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Generation failed"
    exit 1
fi

echo ""
echo "✓ Generation complete!"
echo ""

# Step 2: Judge responses
echo "Step 2: Judging responses..."
python judge_responses.py

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Judging failed"
    exit 1
fi

echo ""
echo "========================================"
echo "✓ Frustration steering experiment completed!"
echo "Responses: outputs/steered_responses.jsonl"
echo "Judgments: outputs/judgments.jsonl"
echo "========================================"
