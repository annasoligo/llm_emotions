#!/bin/bash
#SBATCH --job-name=logit_test
#SBATCH --output=/workspace-vast/annas/git/research-tools/emotion_logit_lens/logs/test_integration_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/emotion_logit_lens/logs/test_integration_%A.err
#SBATCH --time=00:30:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

set -e

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

# Ensure clean GPU state
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

echo "=============================================================================="
echo "TESTING LOGIT LENS INTEGRATION"
echo "=============================================================================="
echo ""

# Run the test script
python emotion_logit_lens/scripts/test_integration.py

echo ""
echo "=============================================================================="
echo "TEST COMPLETE"
echo "=============================================================================="
