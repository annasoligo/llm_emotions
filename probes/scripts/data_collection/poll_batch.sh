#!/bin/bash
# Poll batch API status and retrieve results when complete
#
# Usage: ./poll_batch.sh <batch_id>

BATCH_ID="${1:-msgbatch_0167akXBVtUZrX5qseAQXR5S}"

echo "Polling batch: $BATCH_ID"
echo ""

cd /workspace-vast/annas/git/research-tools

# Load secrets
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate venv
source .venv/bin/activate

# Run polling
python probes/scripts/data_collection/generate_axis_paraphrases_batch.py \
    --batch-id "$BATCH_ID" \
    --output-dir probes/data/axis_paraphrases \
    --poll-interval 120
