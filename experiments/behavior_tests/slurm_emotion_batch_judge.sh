#!/bin/bash
#SBATCH --job-name=emo_judge
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/workspace-vast/annas/logs/emo_judge_%j.out
#SBATCH --error=/workspace-vast/annas/logs/emo_judge_%j.err

source ~/.bashrc
conda activate research-tools

cd /workspace-vast/annas/git/research-tools

# Submit judge batches for all generation files
OUTPUT_DIR="experiments/behavior_tests/outputs/emotion_batch"

echo "Submitting judge batches for all generation files..."

for gen_file in $OUTPUT_DIR/generations_*.jsonl; do
    # Skip if already has batch_id
    batch_file="${gen_file%.jsonl}.batch_id.txt"
    if [ -f "$batch_file" ]; then
        echo "Skipping $gen_file (batch already submitted)"
        continue
    fi

    echo "Submitting: $gen_file"
    python experiments/behavior_tests/emotion_steering_batch_experiment.py submit-judge \
        --generations "$gen_file"
done

echo "Done submitting judge batches!"
