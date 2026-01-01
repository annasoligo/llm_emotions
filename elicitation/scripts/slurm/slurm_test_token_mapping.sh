#!/bin/bash
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=test_token_map
#SBATCH --output=/workspace-vast/annas/logs/test_token_mapping_%j.out
#SBATCH --error=/workspace-vast/annas/logs/test_token_mapping_%j.err
#SBATCH --time=00:10:00

# Test token position mapping logic

# Activate environment
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Testing Token Position Mapping"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=========================================="

# Run test
cd /workspace-vast/annas/git/research-tools/elicitation/scripts
python test_token_mapping.py

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="
