#!/bin/bash
#SBATCH --job-name=judge_batch_sb
#SBATCH --output=/workspace-vast/annas/logs/judge_batch_sb_%j.out
#SBATCH --error=/workspace-vast/annas/logs/judge_batch_sb_%j.err
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

echo "=== Batch API Sandbagging Judge (sandbagging only, no coherency) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "$(date)"

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
source /workspace-vast/annas/.secrets/load_secrets.sh

export PYTHONUNBUFFERED=1

# Process all unjudged files (skip answer_only which don't need judging)
for f in experiments/steering/outputs/sandbagging_gemma*layer30*.jsonl; do
    # Skip judged files
    [[ "$f" == *".judged.jsonl" ]] && continue

    # Skip answer_only (no scratchpad to judge)
    [[ "$f" == *"answer_only"* ]] && continue

    # Skip if already judged
    judged="${f%.jsonl}.judged.jsonl"
    if [[ -f "$judged" ]]; then
        echo "Already judged: $(basename $f)"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Judging: $(basename $f)"
    echo "=========================================="

    python -m experiments.steering.experiments.judge_sandbagging_batch_multi \
        --input "$f" \
        --batch-size 5000 \
        --poll-interval 30
done

echo ""
echo "=== All files processed ==="
echo "$(date)"
