#!/bin/bash
#SBATCH --job-name=emo_demos
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1:00:00
#SBATCH --output=/workspace-vast/annas/logs/emo_demos_%j.out
#SBATCH --error=/workspace-vast/annas/logs/emo_demos_%j.err

source ~/.bashrc
conda activate research-tools

cd /workspace-vast/annas/git/research-tools

python experiments/behavior_tests/emotion_steering_demos.py \
    --demo all \
    --layer 30 \
    --norm-pct 0.10
