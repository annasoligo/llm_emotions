#!/bin/bash
#SBATCH --job-name=ua_emotion_all_layers
#SBATCH --output=/workspace-vast/annas/logs/ua_emotion_all_layers_%j.out
#SBATCH --error=/workspace-vast/annas/logs/ua_emotion_all_layers_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G

# Activate environment
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

cd /workspace-vast/annas/git/research-tools

echo "=================================================="
echo "UA Emotion Disentanglement - All Layers Collection"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=================================================="

# Collect activations at all layers (0-61) with streaming writes
python3 probes/ua_emotion_disentangle/collect_full.py \
    --layers 0-61 \
    --output probes/ua_emotion_disentangle/data/activations/full_all_layers.h5

exit_code=$?

echo "=================================================="
echo "Finished: $(date)"
echo "Exit code: $exit_code"
echo "=================================================="

exit $exit_code
