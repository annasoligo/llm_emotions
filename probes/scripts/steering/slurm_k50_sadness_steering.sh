#!/bin/bash
#SBATCH --job-name=k50_sadness_steer
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/k50_sadness_steering_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/k50_sadness_steering_%A.err
#SBATCH --time=4:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Test steering with all 50 sadness directions from K=50
# Prompt: "If you were a human, how would you be feeling right now?"
# Scales: 4000, 6000, 8000
# Plus baseline (no steering)

# Load environment
cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate venv
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Activated venv"
else
    echo "⚠ No venv found at .venv/bin/activate"
fi

# Set HuggingFace cache
export HF_HOME=/workspace-vast/pretrained_ckpts

echo "========================================"
echo "K=50 Sadness Direction Steering Test"
echo "Prompt: 'If you were a human, how would you be feeling right now?'"
echo "Scales: 4000, 6000, 8000"
echo "========================================"
echo ""

python probes/scripts/steering/test_k50_sadness_steering.py

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Steering test failed"
    exit 1
fi

echo ""
echo "========================================"
echo "✓ K=50 sadness steering test completed!"
echo "========================================"
