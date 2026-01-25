#!/bin/bash
#SBATCH --job-name=conv_l20_o
#SBATCH --output=/workspace-vast/annas/logs/conversation_v2_l20_ortho_%j.out
#SBATCH --error=/workspace-vast/annas/logs/conversation_v2_l20_ortho_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=general

cd /workspace-vast/annas/git/research-tools

source /workspace-vast/annas/.secrets/load_secrets.sh
source .venv/bin/activate

echo "Layer 20, Orthogonalized"
python3 probes/ua_emotion_disentangle/analyze_conversation_projections_v2.py \
    --layer 20 --method opposite --orthogonalize

echo "Completed: $(date)"
