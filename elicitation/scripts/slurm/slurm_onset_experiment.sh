#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --job-name=emotion_onset_exp
#SBATCH --output=/workspace-vast/annas/logs/emotion_onset_exp_%j.out
#SBATCH --error=/workspace-vast/annas/logs/emotion_onset_exp_%j.err
#SBATCH --time=04:00:00

# Run emotion onset analysis experiment
# Extracts activations at windows around emotion onset positions

# Activate environment
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "Starting emotion onset experiment..."
cd /workspace-vast/annas/git/research-tools/elicitation/scripts

# Run experiment (pass optional --limit N argument)
if [ -n "$1" ]; then
    echo "Processing first $1 samples"
    python run_emotion_onset_experiment.py --limit $1
else
    echo "Processing all samples"
    python run_emotion_onset_experiment.py
fi
