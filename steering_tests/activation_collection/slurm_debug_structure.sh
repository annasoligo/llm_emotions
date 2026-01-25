#!/bin/bash
#SBATCH --job-name=debug_structure
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=0:30:00
#SBATCH --output=logs/debug_structure_%j.out
#SBATCH --error=logs/debug_structure_%j.err

# Activate virtual environment
source .venv/bin/activate

# Load secrets
source /workspace-vast/annas/.secrets/load_secrets.sh

# Run debug script
python steering_tests/activation_collection/debug_model_structure.py
