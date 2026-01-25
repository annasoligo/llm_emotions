#!/bin/bash
#SBATCH --job-name=judge_L2
#SBATCH --output=/workspace-vast/annas/logs/judge_last2_%j.out
#SBATCH --error=/workspace-vast/annas/logs/judge_last2_%j.err
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00

# Judge results from last-2-layers anti-steer experiment
# Runs both sandbagging+coherency and fear sentiment judges

set -e

source /workspace-vast/annas/.secrets/load_secrets.sh

echo "=== Judging Last 2 Layers Anti-Steer Results ==="
date

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Find the most recent textmeandiff file with L60-61 in the path or recent timestamp
INPUT_FILE=$(ls -t experiments/steering/outputs/anti_steer/sandbagging_multilayer_antisteer_*_textmeandiff.jsonl 2>/dev/null | head -1)

if [ -z "$INPUT_FILE" ]; then
    echo "No input file found!"
    exit 1
fi

echo "Input file: $INPUT_FILE"
echo "Samples: $(wc -l < $INPUT_FILE)"

echo ""
echo "=== Running Sandbagging + Coherency Judge ==="
python -m experiments.steering.experiments.judge_sandbagging_coherency_async \
    --input "$INPUT_FILE" \
    --concurrency 50

# Get the judged file path
JUDGED_FILE="${INPUT_FILE%.jsonl}.judged.jsonl"

echo ""
echo "=== Running Fear Sentiment Judge ==="
python -m experiments.steering.experiments.judge_fear_sentiment_async \
    --input "$JUDGED_FILE" \
    --concurrency 50

echo ""
echo "All judging complete!"
date
