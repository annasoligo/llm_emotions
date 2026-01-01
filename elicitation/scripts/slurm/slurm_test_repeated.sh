#!/bin/bash
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=test_repeated
#SBATCH --output=/workspace-vast/annas/logs/test_repeated_%j.out
#SBATCH --error=/workspace-vast/annas/logs/test_repeated_%j.err
#SBATCH --time=00:05:00

# Test repeated turn content

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "Testing repeated turn content..."
cd /workspace-vast/annas/git/research-tools/elicitation/scripts
python test_repeated_turns.py
