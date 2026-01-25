#!/bin/bash
#SBATCH --job-name=add_logit_lens
#SBATCH --output=/workspace-vast/annas/logs/add_logit_lens_%j.log
#SBATCH --error=/workspace-vast/annas/logs/add_logit_lens_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

echo "Job started at $(date)"
echo "Running on node: $(hostname)"

# Change to project directory
cd /workspace-vast/annas/git/research-tools/eval_dashboard

# Load secrets
if [ -f /workspace-vast/annas/.secrets ]; then
    source /workspace-vast/annas/.secrets
    echo "✓ Loaded secrets"
else
    echo "Warning: No secrets file found"
fi

# Activate venv
if [ -f /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate ]; then
    source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate
    echo "✓ Activated .venv"
else
    echo "ERROR: venv not found"
    exit 1
fi

# Show environment info
echo "Python: $(which python3)"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"

# Run preprocessing script
echo ""
echo "="*80
echo "Recomputing logit lens (emotion) scores with centering..."
echo "="*80
python3 add_logit_lens_to_existing.py

echo ""
echo "Job completed at $(date)"
