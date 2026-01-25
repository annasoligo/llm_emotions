#!/bin/bash
#SBATCH --job-name=ua_layer_sel
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/annas/logs/ua_layer_selection_%j.out
#SBATCH --error=/workspace-vast/annas/logs/ua_layer_selection_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

cd /workspace-vast/annas/git/research-tools

echo "=================================================="
echo "UA Layer Selection for Emotion Steering Vectors"
echo "=================================================="
echo ""
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# Run for first_asst_token position (most relevant for steering)
echo "Computing metrics for first_asst_token..."
python -m experiments.layer_selection.compute_ua_layer_metrics \
    --h5-path probes/ua_emotion_disentangle/data/gemma3_27b_ua_emotions_v4_alllayers.h5 \
    --output-dir experiments/layer_selection/results/ua/ \
    --token-position first_asst_token \
    --model google/gemma-3-27b-it \
    --n-random 100 \
    --pca-k 50

echo ""
echo "Generating plots for first_asst_token..."
python -m experiments.layer_selection.plot_ua_layer_metrics \
    --results experiments/layer_selection/results/ua/ua_layer_metrics_first_asst_token.npz \
    --output-dir experiments/layer_selection/results/ua/

# Also run for last_user_token to compare
echo ""
echo "Computing metrics for last_user_token..."
python -m experiments.layer_selection.compute_ua_layer_metrics \
    --h5-path probes/ua_emotion_disentangle/data/gemma3_27b_ua_emotions_v4_alllayers.h5 \
    --output-dir experiments/layer_selection/results/ua/ \
    --token-position last_user_token \
    --model google/gemma-3-27b-it \
    --n-random 100 \
    --pca-k 50

echo ""
echo "Generating plots for last_user_token..."
python -m experiments.layer_selection.plot_ua_layer_metrics \
    --results experiments/layer_selection/results/ua/ua_layer_metrics_last_user_token.npz \
    --output-dir experiments/layer_selection/results/ua/

echo ""
echo "=================================================="
echo "Done!"
echo "End time: $(date)"
echo "=================================================="
