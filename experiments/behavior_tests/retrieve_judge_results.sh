#!/bin/bash
# Retrieve judge results for all generation files
# Run this after batches have completed (usually ~30min to a few hours)

source ~/.bashrc
conda activate research-tools

cd /workspace-vast/annas/git/research-tools

OUTPUT_DIR="experiments/behavior_tests/outputs/emotion_batch"

echo "Retrieving judge results..."

for gen_file in $OUTPUT_DIR/generations_*.jsonl; do
    # Only process if has batch_id but no judged file
    batch_file="${gen_file%.jsonl}.batch_id.txt"
    judged_file="${gen_file%.jsonl}.judged.jsonl"

    if [ ! -f "$batch_file" ]; then
        echo "Skipping $gen_file (no batch submitted)"
        continue
    fi

    if [ -f "$judged_file" ]; then
        echo "Skipping $gen_file (already judged)"
        continue
    fi

    echo "Retrieving: $gen_file"
    python experiments/behavior_tests/emotion_steering_batch_experiment.py retrieve-judge \
        --generations "$gen_file"
done

echo "Done!"
