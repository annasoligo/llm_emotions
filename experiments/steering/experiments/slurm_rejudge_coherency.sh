#!/bin/bash
#SBATCH --job-name=rejudge_coherency
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=1:00:00
#SBATCH --output=/workspace-vast/annas/logs/rejudge_coherency_%j.out
#SBATCH --error=/workspace-vast/annas/logs/rejudge_coherency_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

cd /workspace-vast/annas/git/research-tools

echo "Re-judging coherency scores with improved prompt..."
echo ""

python experiments/steering/experiments/rejudge_coherency.py --layers 20 25 30 35 40 45

echo ""
echo "Done!"
