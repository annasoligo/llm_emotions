#!/bin/bash
#SBATCH --job-name=analyze_first_tok
#SBATCH --output=/workspace-vast/annas/logs/analyze_first_tok_%j.out
#SBATCH --error=/workspace-vast/annas/logs/analyze_first_tok_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

cd /workspace-vast/annas/git/research-tools/probes/ua_emotion_disentangle

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

python3 analyze_first_tokens_only.py
