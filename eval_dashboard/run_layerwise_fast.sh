#!/bin/bash
#SBATCH --job-name=layerwise_fast
#SBATCH --output=logs/layerwise_fast_%j.log
#SBATCH --error=logs/layerwise_fast_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00

echo "================================================================================"
echo "FAST LAYERWISE PREPROCESSING (150x speedup with subset projection)"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

cd /workspace-vast/annas/git/research-tools/eval_dashboard

DATASETS=(
    "high_emotion_6plus_with_axes.pkl"
    "mid_emotion_3to5_with_axes.pkl"
    "low_emotion_0to2_with_axes.pkl"
    "low_emotion_with_shutdown_with_axes.pkl"
    "high_emotion_with_shutdown_with_axes.pkl"
    "baseline_v12_solvable_with_axes.pkl"
)

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "================================================================================"
    echo "Processing: $dataset"
    echo "================================================================================"
    
    python add_logit_lens_to_existing_BATCHED.py \
        --input "data/$dataset" \
        --output "data/$dataset"
done

echo ""
echo "================================================================================"
echo "DONE! All datasets processed."
echo "End time: $(date)"
echo "================================================================================"
