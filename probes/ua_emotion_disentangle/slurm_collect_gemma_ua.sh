#!/bin/bash
#SBATCH --job-name=gemma_ua
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=/workspace-vast/annas/logs/gemma_ua_%j.out
#SBATCH --error=/workspace-vast/annas/logs/gemma_ua_%j.err

source /workspace-vast/annas/git/research-tools/.venv/bin/activate
source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools

echo "============================================================"
echo "GEMMA UA EMOTION COLLECTION (FULL M×U COMBINATIONS)"
echo "============================================================"

python probes/ua_emotion_disentangle/collect_gemma_corrected.py \
    --layers 0-61 \
    --output probes/ua_emotion_disentangle/data/gemma3_27b_ua_emotions_v4_alllayers.h5

echo ""
echo "Done!"
