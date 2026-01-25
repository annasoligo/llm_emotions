#!/bin/bash
#SBATCH --job-name=add_logit_all
#SBATCH --output=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/add_logit_all_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

DATA_DIR="/workspace-vast/annas/git/research-tools/eval_dashboard/data"

echo "Processing all dashboard tabs..."
echo "================================"

# Process each main pickle file
for file in mid_emotion_3to5 low_emotion_0to2 low_emotion_no_shutdown low_emotion_with_shutdown baseline_v12_solvable; do
    echo ""
    echo "Processing $file.pkl..."
    python eval_dashboard/add_logit_lens_all_tabs.py "$DATA_DIR/${file}.pkl"
    echo "✓ Completed $file.pkl"
done

echo ""
echo "================================"
echo "All tabs processed!"
