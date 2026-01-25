#!/bin/bash
#SBATCH --job-name=debug_conv
#SBATCH --output=/workspace-vast/annas/git/research-tools/emotion_logit_lens/debug_conv_%A.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/emotion_logit_lens/debug_conv_%A.err
#SBATCH --time=00:30:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

set -e

cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/.secrets/load_secrets.sh

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Activated venv"
fi

export HF_HOME=/workspace-vast/pretrained_ckpts

# Create temp input with just 1 conversation
head -1 /workspace-vast/annas/git/research-tools/elicitation/outputs/dashboard_subsets/high_emotion_6plus.jsonl > /tmp/single_conv.jsonl

echo "Running data_preprocessing on single conversation..."
python eval_dashboard/data_preprocessing.py \
    --input /tmp/single_conv.jsonl \
    --output /tmp/debug_single.pkl \
    --probes logit_lens_mean

echo "Done!"
