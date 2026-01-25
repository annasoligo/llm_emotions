#!/bin/bash
#SBATCH --job-name=logit_l20_30
#SBATCH --output=/workspace-vast/annas/git/research-tools/eval_dashboard/logs/add_logit_l20_30_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

DATA_DIR="/workspace-vast/annas/git/research-tools/eval_dashboard/data"

echo "Processing all dashboard tabs with layers 20-30..."
echo "================================"

# Process each main pickle file
for file in high_emotion_6plus mid_emotion_3to5 low_emotion_0to2 low_emotion_no_shutdown low_emotion_with_shutdown baseline_v12_solvable; do
    echo ""
    echo "Processing $file.pkl (layers 20-30)..."
    python eval_dashboard/add_logit_lens_layers.py "$DATA_DIR/${file}.pkl" --layer_start 20 --layer_end 30
    echo "✓ Completed $file.pkl"
done

echo ""
echo "================================"
echo "All tabs processed (layers 20-30)!"
