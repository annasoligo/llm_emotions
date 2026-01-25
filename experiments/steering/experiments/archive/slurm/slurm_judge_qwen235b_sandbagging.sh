#!/bin/bash
#SBATCH --job-name=judge_sb235b
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/annas/logs/judge_sb235b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/judge_sb235b_%j.err

# Judge sandbagging results with both sandbagging and coherency judges
# Uses Anthropic Batch API

# Load environment
source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

cd /workspace-vast/annas/git/research-tools

echo "============================================================"
echo "JUDGING SANDBAGGING RESULTS - QWEN 235B"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Input: $INPUT_FILE"
echo "============================================================"

if [ -z "$INPUT_FILE" ]; then
    echo "ERROR: INPUT_FILE environment variable not set"
    echo "Usage: INPUT_FILE=path/to/results.jsonl sbatch slurm_judge_qwen235b_sandbagging.sh"
    exit 1
fi

python -m experiments.steering.experiments.judge_sandbagging_coherency_batch \
    --input "$INPUT_FILE" \
    --poll-interval 30

echo ""
echo "============================================================"
echo "JUDGING COMPLETE"
echo "============================================================"
