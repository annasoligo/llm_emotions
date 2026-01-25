#!/bin/bash
#SBATCH --job-name=emo_decay_replot
#SBATCH --output=/workspace-vast/annas/logs/emo_decay_replot_%j.out
#SBATCH --error=/workspace-vast/annas/logs/emo_decay_replot_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

echo "=================================================="
echo "Emotion Decay Replot"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=================================================="

cd /workspace-vast/annas/git/research-tools/probes/ua_emotion_disentangle

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

python3 replot_emotion_decay.py

echo "=================================================="
echo "Completed: $(date)"
echo "=================================================="
