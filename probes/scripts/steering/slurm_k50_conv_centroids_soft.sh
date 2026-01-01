#!/bin/bash
#SBATCH --job-name=k50_conv_soft
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/k50_conv_centroids_soft_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/k50_conv_centroids_soft_%A.err
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

cd /workspace-vast/annas/git/research-tools

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Activated venv"
fi

echo "========================================"
echo "K=50 Conversation Centroids (SOFT)"
echo "4 prompts × 6 emotions × 2 roles"
echo "Scale: 5000"
echo "========================================"

python probes/scripts/steering/test_k50_conversation_centroids_steering.py --constraint soft

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Steering experiment failed"
    exit 1
fi

echo ""
echo "========================================"
echo "✓ K=50 soft constraint steering completed!"
echo "========================================"
