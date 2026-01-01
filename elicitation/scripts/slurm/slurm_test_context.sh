#!/bin/bash
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=test_context
#SBATCH --output=/workspace-vast/annas/logs/test_context_%j.out
#SBATCH --error=/workspace-vast/annas/logs/test_context_%j.err
#SBATCH --time=00:10:00

# Test context-based matching

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "Testing context-based word matching..."
cd /workspace-vast/annas/git/research-tools/elicitation/scripts
python test_context_matching.py
