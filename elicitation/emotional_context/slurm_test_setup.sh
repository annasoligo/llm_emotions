#!/bin/bash
#SBATCH --job-name=test_emotional_context
#SBATCH --output=logs/test_setup_%j.log
#SBATCH --error=logs/test_setup_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# Load authentication (includes API keys)
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate research-tools venv
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

# Set up environment variables
export PYTHONPATH=/workspace-vast/annas/git/believe-it-or-not:$PYTHONPATH

# Run test
cd /workspace-vast/annas/git/research-tools/elicitation/emotional_context
python test_setup.py
