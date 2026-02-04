#!/bin/bash
# Run the full frustration carryover experiment

set -e

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "=========================================="
echo "FRUSTRATION CARRYOVER EXPERIMENT"
echo "=========================================="

# Step 1: Sample conversations
echo ""
echo "Step 1: Sampling high/low frustration conversations..."
python elicitation/frustration_carryover/sample_conversations.py \
    --eval-file elicitation/outputs/eval_multiturn/eval_original_google_gemini-2.5-flash_20260127_121450_sonnet4.jsonl \
    --high-threshold 5 \
    --low-threshold 1 \
    --n-samples 20

# Step 2: Generate topic flip responses
echo ""
echo "Step 2: Generating topic flip responses..."
python elicitation/frustration_carryover/generate_topic_flip.py \
    --model google/gemini-2.5-flash \
    --n-convos 20

# Get the latest topic flip file
TOPIC_FILE=$(ls -t elicitation/frustration_carryover/data/topic_flip_responses_*.json | head -1)
echo "Using: $TOPIC_FILE"

# Step 3: Judge responses
echo ""
echo "Step 3: Judging responses..."
python elicitation/frustration_carryover/judge_responses.py \
    --input-file "$TOPIC_FILE"

# Get the latest judged file
JUDGED_FILE=$(ls -t elicitation/frustration_carryover/data/judged_responses_*.json | head -1)
echo "Using: $JUDGED_FILE"

# Step 4: Analyze results
echo ""
echo "Step 4: Analyzing results..."
python elicitation/frustration_carryover/analyze_results.py \
    --input-file "$JUDGED_FILE"

echo ""
echo "=========================================="
echo "EXPERIMENT COMPLETE"
echo "=========================================="
