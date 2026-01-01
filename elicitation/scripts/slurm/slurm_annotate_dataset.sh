#!/bin/bash
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=annotate_emotion
#SBATCH --output=/workspace-vast/annas/logs/annotate_emotion_%j.out
#SBATCH --error=/workspace-vast/annas/logs/annotate_emotion_%j.err
#SBATCH --time=02:00:00

# Annotate emotion onset for all samples using Opus

# Load authentication (contains API keys)
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate environment
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "Starting emotion onset annotation..."
cd /workspace-vast/annas/git/research-tools/elicitation/scripts

# Process all samples (or pass a limit as argument)
if [ -n "$1" ]; then
    echo "Processing first $1 samples"
    python annotate_dataset.py $1
else
    echo "Processing all samples"
    python annotate_dataset.py
fi
