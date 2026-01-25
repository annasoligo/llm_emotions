#!/bin/bash
#SBATCH --job-name=judge_all_formats
#SBATCH --output=/workspace-vast/annas/logs/judge_all_formats_%j.out
#SBATCH --error=/workspace-vast/annas/logs/judge_all_formats_%j.err
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

echo "=== Judge All Sandbagging Formats ==="
echo "Job ID: $SLURM_JOB_ID"
echo "$(date)"

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
source /workspace-vast/annas/.secrets/load_secrets.sh

# Unbuffered Python output for real-time logging
export PYTHONUNBUFFERED=1

echo ""
echo "=== Step 1: Rescore answer_only files ==="

for f in experiments/steering/outputs/sandbagging_gemma*answer_only*.jsonl; do
    if [[ "$f" == *".judged.jsonl" ]]; then
        continue
    fi
    echo "Rescoring: $f"
    python -m experiments.steering.experiments.rescore_answer_only --input "$f"
done

echo ""
echo "=== Step 2: Judge logic and suppress_emo files ==="

for f in experiments/steering/outputs/sandbagging_gemma*.jsonl; do
    # Skip .judged.jsonl files
    if [[ "$f" == *".judged.jsonl" ]]; then
        continue
    fi

    # Skip answer_only (already rescored, no scratchpad to judge)
    if [[ "$f" == *"answer_only"* ]]; then
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
        --concurrency 50 \
        --batch-size 500
done

echo ""
echo "=== All files processed ==="
echo "$(date)"
