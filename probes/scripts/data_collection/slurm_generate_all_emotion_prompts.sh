#!/bin/bash
#SBATCH --job-name=gen_all_emo
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/gen_all_emo_%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/gen_all_emo_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Change to project directory
cd /workspace-vast/annas/git/research-tools

# Load secrets
echo "Loading secrets..."
source /workspace-vast/annas/.secrets/load_secrets.sh
echo "✓ Secrets loaded"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Parse arguments
MODE="${1:-test}"  # test or full
SAMPLES="${2:-5}"  # samples per emotion

if [ "$MODE" == "test" ]; then
    echo "=========================================="
    echo "Running TEST mode"
    echo "Samples per emotion: $SAMPLES"
    echo "Topics: 2"
    echo "=========================================="
    echo ""

    python probes/scripts/data_collection/generate_all_emotion_prompts.py \
        --output probes/data/emotion_prompts_test.jsonl \
        --samples_per_emotion "$SAMPLES" \
        --topics 2 \
        --strategies_cache probes/data/emotion_strategies_cache.json \
        --max_concurrent 5

elif [ "$MODE" == "full" ]; then
    echo "=========================================="
    echo "Running FULL mode"
    echo "Samples per emotion: $SAMPLES"
    echo "Topics: all"
    echo "=========================================="
    echo ""

    python probes/scripts/data_collection/generate_all_emotion_prompts.py \
        --output probes/data/emotion_prompts_${SAMPLES}.jsonl \
        --samples_per_emotion "$SAMPLES" \
        --strategies_cache probes/data/emotion_strategies_cache.json \
        --max_concurrent 10
else
    echo "Unknown mode: $MODE"
    echo "Usage: sbatch slurm_generate_all_emotion_prompts.sh [test|full] [samples_per_emotion]"
    exit 1
fi

exit_code=$?

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code
