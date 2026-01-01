#!/bin/bash
#SBATCH --job-name=test_annotation
#SBATCH --output=/workspace-vast/annas/git/research-tools/elicitation/outputs/test_annotation_%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/elicitation/outputs/test_annotation_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu

# Activate your venv (adjust path as needed)
source /workspace-vast/annas/venvs/your_venv/bin/activate  # CHANGE THIS

# Set API key from environment or file
export ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}

# Run the annotation script
cd /workspace-vast/annas/git/research-tools/elicitation/scripts
python annotate_emotion_onset.py
