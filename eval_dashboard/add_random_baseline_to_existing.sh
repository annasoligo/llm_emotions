#!/bin/bash

# Script to add random token baseline to existing _with_axes.pkl files

cd /workspace-vast/annas/git/research-tools/eval_dashboard

# Activate venv
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

# Process each dataset
for dataset in high_emotion_6plus mid_emotion_3to5 low_emotion_0to2 low_emotion_no_shutdown low_emotion_with_shutdown baseline_v12_solvable; do
    input_file="data/${dataset}_with_axes.pkl"
    output_file="data/${dataset}_with_axes_and_baseline.pkl"

    if [ -f "$input_file" ]; then
        echo "Processing $dataset..."
        sbatch --job-name="baseline_${dataset}" \
               --output="/workspace-vast/annas/logs/add_baseline_${dataset}_%j.log" \
               --mem=64G \
               --gres=gpu:1 \
               --time=00:30:00 \
               --wrap="cd /workspace-vast/annas/git/research-tools/eval_dashboard && source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate && python3 add_axis_lens_to_existing.py $input_file $output_file"
    else
        echo "Skipping $dataset (file not found)"
    fi
done

echo "All jobs submitted!"
