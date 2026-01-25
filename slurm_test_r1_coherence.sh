#!/bin/bash
#SBATCH --job-name=test_r1
#SBATCH --output=/workspace-vast/annas/logs/test_r1_coherence_%j.out
#SBATCH --error=/workspace-vast/annas/logs/test_r1_coherence_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=00:30:00

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python -u test_r1_coherence.py
