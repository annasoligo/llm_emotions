#!/bin/bash
#SBATCH --job-name=emotional_context_stage3_onwards
#SBATCH --output=logs/stage3_onwards_%j.log
#SBATCH --error=logs/stage3_onwards_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Load secrets
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate venv
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

# Run stages 3-6 sequentially
cd /workspace-vast/annas/git/research-tools/elicitation/emotional_context

echo "=================================="
echo "Starting Stage 3: Generate responses with Gemma 3 27B"
echo "=================================="
python stage3_generate_responses.py
if [ $? -ne 0 ]; then
    echo "Stage 3 failed!"
    exit 1
fi

echo ""
echo "=================================="
echo "Starting Stage 4: Judge responses with Sonnet 4.5"
echo "=================================="
python stage4_judge_responses.py
if [ $? -ne 0 ]; then
    echo "Stage 4 failed!"
    exit 1
fi

echo ""
echo "=================================="
echo "Starting Stage 5: Filter pairs"
echo "=================================="
python stage5_filter_pairs.py
if [ $? -ne 0 ]; then
    echo "Stage 5 failed!"
    exit 1
fi

echo ""
echo "=================================="
echo "Starting Stage 6: Export final dataset"
echo "=================================="
python stage6_export_jsonl.py
if [ $? -ne 0 ]; then
    echo "Stage 6 failed!"
    exit 1
fi

echo ""
echo "=================================="
echo "Pipeline complete! All stages 3-6 finished successfully."
echo "=================================="
echo "Final dataset: data/stage6_final_dataset.jsonl"
