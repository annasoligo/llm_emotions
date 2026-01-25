#!/bin/bash
#SBATCH --job-name=judge_50pct
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/annas/logs/judge_50pct_%j.out
#SBATCH --error=/workspace-vast/annas/logs/judge_50pct_%j.err

# Judge sandbagging results for 50% text-based steering

# Load secrets and environment
source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

cd /workspace-vast/annas/git/research-tools

echo "Judging 50% steering conditions..."
echo "Conditions: baseline, fear_+/-50%, happiness_+/-50%, anger_+/-50%"

python experiments/behavior_tests/judge_sandbagging_batch.py \
    --results experiments/steering/outputs/sandbagging_qwen235b_text_fear_happiness_anger_layer45_20260115_132434.jsonl \
    --conditions baseline "fear_+50%" "fear_-50%" "happiness_+50%" "happiness_-50%" "anger_+50%" "anger_-50%" \
    --output experiments/steering/outputs/judged_text_L45_50pct.jsonl \
    --poll-interval 30

echo "Done!"
