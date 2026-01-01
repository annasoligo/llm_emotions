#!/bin/bash
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=test_real_data
#SBATCH --output=/workspace-vast/annas/logs/test_real_data_%j.out
#SBATCH --error=/workspace-vast/annas/logs/test_real_data_%j.err
#SBATCH --time=00:10:00

# Test context matching on real data

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "Testing context matching on real samples..."
cd /workspace-vast/annas/git/research-tools/elicitation/scripts
python test_real_data.py
