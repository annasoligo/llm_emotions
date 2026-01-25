#!/bin/bash
#SBATCH --job-name=olmo_ua
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=/workspace-vast/annas/logs/olmo_ua_%j.out
#SBATCH --error=/workspace-vast/annas/logs/olmo_ua_%j.err

source /workspace-vast/annas/git/research-tools/.venv/bin/activate
cd /workspace-vast/annas/git/research-tools

echo "============================================================"
echo "OLMO UA EMOTION COLLECTION (CORRECTED - BOTH DIRECTIONS)"
echo "============================================================"

python probes/ua_emotion_disentangle/collect_olmo_corrected.py \
    --layers 0-63 \
    --output probes/ua_emotion_disentangle/data/olmo_32b_ua_emotions_v4_alllayers.h5

echo ""
echo "Done!"
