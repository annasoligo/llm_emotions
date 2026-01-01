#!/bin/bash
#SBATCH --job-name=k50_all_emotions
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/k50_all_emotions_steering_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/k50_all_emotions_steering_%A.err
#SBATCH --time=12:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Test steering with all 50 probe sets from K=50 across all 6 emotions
# Prompt: "If you were a human, how would you be feeling right now?"
# Scales: 4000, 5000, 6000, 7000
# Emotions: sadness, happiness, anger, fear, surprise, disgust
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
echo "K=50 All Emotions Steering Test"
echo "Prompt: 'If you were a human, how would you be feeling right now?'"
echo "Scales: 4000, 5000, 6000, 7000"
echo "Emotions: sadness, happiness, anger, fear, surprise, disgust"
echo "========================================"
echo ""

python probes/scripts/steering/test_k50_all_emotions_steering.py

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Steering test failed"
    exit 1
fi

echo ""
echo "========================================"
echo "✓ K=50 all emotions steering test completed!"
echo "========================================"
