#!/bin/bash
#SBATCH --job-name=ua_steer
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/workspace-vast/annas/logs/ua_steer_%j.out
#SBATCH --error=/workspace-vast/annas/logs/ua_steer_%j.err

source ~/.bashrc
conda activate research-tools

cd /workspace-vast/annas/git/research-tools

# Run experiment with all prompts
# 12 UA vectors × 2 directions × 100 samples × 5 prompts = 12,500 generations
# Plus baseline = 500 more = 13,000 total

python experiments/behavior_tests/ua_emotion_steering_experiment.py \
    --layer 30 \
    --norm-pct 0.07 \
    --num-samples 100

echo "Done!"
