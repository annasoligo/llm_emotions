#!/bin/bash
#SBATCH --job-name=olmo_texts
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/workspace-vast/annas/logs/olmo_texts_%j.out
#SBATCH --error=/workspace-vast/annas/logs/olmo_texts_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

cd /workspace-vast/annas/git/research-tools

echo "============================================================"
echo "OLMO TEXT ACTIVATION COLLECTION"
echo "============================================================"
echo "Model: allenai/OLMo-3-32B-Think"
echo "Layers: 22-42 (middle 21 layers)"
echo ""

python probes/ua_emotion_disentangle/collect_texts_olmo.py \
    --model "allenai/OLMo-3-32B-Think" \
    --layers "22-42" \
    --output "probes/ua_emotion_disentangle/data/olmo_32b_texts_combined.h5"

echo ""
echo "Done!"
