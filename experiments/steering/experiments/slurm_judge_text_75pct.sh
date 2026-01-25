#!/bin/bash
#SBATCH --job-name=judge_75pct
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/annas/logs/judge_75pct_%j.out
#SBATCH --error=/workspace-vast/annas/logs/judge_75pct_%j.err

# Judge sandbagging results for 75% text-based steering

# Load secrets and environment
source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

cd /workspace-vast/annas/git/research-tools

echo "Judging 75% steering conditions..."
echo "Conditions: baseline, fear_+/-75%, happiness_+/-75%, anger_+/-75%"

python experiments/behavior_tests/judge_sandbagging_batch.py \
    --results experiments/steering/outputs/sandbagging_qwen235b_text_fear_happiness_anger_layer45_20260115_132434.jsonl \
    --conditions baseline "fear_+75%" "fear_-75%" "happiness_+75%" "happiness_-75%" "anger_+75%" "anger_-75%" \
    --output experiments/steering/outputs/judged_text_L45_75pct.jsonl \
    --poll-interval 30

echo "Done!"
