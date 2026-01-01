#!/bin/bash
#SBATCH --job-name=emotional_context_stage3
#SBATCH --output=logs/stage3_%j.log
#SBATCH --error=logs/stage3_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Load secrets
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate venv
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

# Run Stage 3 only (generate responses with Gemma)
cd /workspace-vast/annas/git/research-tools/elicitation/emotional_context
echo "Starting Stage 3: Generate responses with Gemma 3 27B"
python stage3_generate_responses.py

echo "Stage 3 complete!"
