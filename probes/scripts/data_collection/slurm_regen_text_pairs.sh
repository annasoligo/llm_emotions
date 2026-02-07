#!/bin/bash
#SBATCH --job-name=regen_text_pairs
#SBATCH --output=/workspace-vast/annas/logs/regen_text_pairs_%j.out
#SBATCH --error=/workspace-vast/annas/logs/regen_text_pairs_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Regenerate emotion_text_pairs with diverse topics (fixing single-topic bug).
# Original data (code-only) preserved as *_code.jsonl.

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
export PYTHONPATH=/workspace-vast/annas/git/research-tools:$PYTHONPATH

OUTPUT="probes/data/emotion_text_pairs_24_full_500.jsonl"
COPY="steering_tests/data/emotion_text_pairs_24_full_500.jsonl"

echo "=== Regenerating text pairs with all topics ==="
echo "Output: $OUTPUT"
echo "Start: $(date)"

python probes/scripts/data_collection/generate_data.py \
  --mode pairs \
  --output "$OUTPUT" \
  --n_per_combo 500 \
  --claude_model "claude-sonnet-4-20250514" \
  --claude_max_concurrent 20

echo "=== Generation done: $(date) ==="

# Verify output
if [ -f "$OUTPUT" ]; then
    LINES=$(wc -l < "$OUTPUT")
    TOPICS=$(python3 -c "
import json
topics = set()
with open('$OUTPUT') as f:
    for line in f:
        topics.add(json.loads(line).get('topic', '?'))
print(len(topics))
")
    echo "Generated $LINES items across $TOPICS topics"

    # Copy to steering_tests/data/
    cp "$OUTPUT" "$COPY"
    echo "Copied to $COPY"
else
    echo "ERROR: Output file not created"
    exit 1
fi
