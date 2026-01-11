#!/bin/bash
#SBATCH --job-name=rejudge_L30
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:30:00
#SBATCH --output=/workspace-vast/annas/logs/rejudge_L30_%j.out
#SBATCH --error=/workspace-vast/annas/logs/rejudge_L30_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

cd /workspace-vast/annas/git/research-tools

echo "Re-judging L30 with format-agnostic prompt..."
echo ""

python experiments/steering/experiments/rejudge_coherency.py --layers 30

echo ""
echo "Done!"
