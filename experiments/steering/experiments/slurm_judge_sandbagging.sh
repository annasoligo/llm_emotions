#!/bin/bash
#SBATCH --job-name=judge_sandbagging
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=1:00:00
#SBATCH --output=/workspace-vast/annas/logs/judge_sandbagging_%j.out
#SBATCH --error=/workspace-vast/annas/logs/judge_sandbagging_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

cd /workspace-vast/annas/git/research-tools

INPUT_FILE=${1:-"experiments/steering/outputs/sandbagging_fear_7pct_layer30.jsonl"}

python experiments/steering/experiments/judge_sandbagging_responses.py "$INPUT_FILE"
