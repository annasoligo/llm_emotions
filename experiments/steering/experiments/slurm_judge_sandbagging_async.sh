#!/bin/bash
#SBATCH --job-name=judge_async
#SBATCH --output=/workspace-vast/annas/logs/judge_async_%j.out
#SBATCH --error=/workspace-vast/annas/logs/judge_async_%j.err
#SBATCH --partition=general
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

echo "=== Async Sandbagging Judge ==="
echo "Job ID: $SLURM_JOB_ID"
echo "$(date)"

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
source /workspace-vast/annas/.secrets/load_secrets.sh

# Find all unjudged files (excluding answer_only which don't need judges)
echo ""
echo "Looking for files to judge..."

for f in experiments/steering/outputs/sandbagging_gemma*.jsonl; do
    # Skip .judged.jsonl files (already processed outputs)
    if [[ "$f" == *".judged.jsonl" ]]; then
        continue
    fi

    # Skip answer_only format
    if [[ "$f" == *"answer_only"* ]]; then
        echo "Skipping answer_only: $f"
        continue
    fi

    # Skip already judged files
    judged_file="${f%.jsonl}.judged.jsonl"
    if [[ -f "$judged_file" ]]; then
        echo "Already judged: $f"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Judging: $f"
    echo "=========================================="

    python -m experiments.steering.experiments.judge_sandbagging_coherency_async \
        --input "$f" \
        --concurrency 20
done

echo ""
echo "=== All files processed ==="
echo "$(date)"
