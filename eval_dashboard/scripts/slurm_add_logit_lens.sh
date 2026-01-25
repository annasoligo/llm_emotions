#!/bin/bash
#SBATCH --job-name=add_logit
#SBATCH --output=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/add_logit_lens_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/add_logit_lens_%A.err
#SBATCH --time=02:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
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

# Run the merge script
python eval_dashboard/scripts/add_logit_lens_to_existing.py

echo ""
echo "Dashboard is ready! Logit lens methods added."
