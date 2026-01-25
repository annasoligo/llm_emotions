#!/bin/bash
#SBATCH --job-name=emo_decay
#SBATCH --output=/workspace-vast/annas/logs/emo_decay_%j.out
#SBATCH --error=/workspace-vast/annas/logs/emo_decay_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

echo "=================================================="
echo "Emotion Decay Measurement Experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=================================================="

source /workspace-vast/annas/git/research-tools/.venv/bin/activate
cd /workspace-vast/annas/git/research-tools/probes/ua_emotion_disentangle

python3 measure_emotion_decay.py

echo "=================================================="
echo "Completed: $(date)"
echo "=================================================="
