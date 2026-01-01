#!/bin/bash
#SBATCH --job-name=k10_conv_hard
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/k10_conv_centroids_hard_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/k10_conv_centroids_hard_%A.err
#SBATCH --time=2:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Test steering with K=10 conversation centroids (HARD constraint / Gram-Schmidt)

cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate venv
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Activated venv"
fi

export HF_HOME=/workspace-vast/pretrained_ckpts

echo "========================================"
echo "K=10 Conversation Centroids (HARD)"
echo "User + Assistant centroids"
echo "2 prompts × 6 emotions × 2 roles × 3 scales"
echo "========================================"
echo ""

python probes/scripts/steering/test_k10_conversation_centroids_steering.py --constraint hard

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Steering test failed"
    exit 1
fi

echo ""
echo "========================================"
echo "✓ K=10 hard constraint steering completed!"
echo "========================================"
