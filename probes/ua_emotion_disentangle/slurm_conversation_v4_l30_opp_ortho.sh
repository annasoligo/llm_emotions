#!/bin/bash
#SBATCH --job-name=v4_l30_opp_o
#SBATCH --output=/workspace-vast/annas/logs/conversation_v4_l30_opp_ortho_%j.out
#SBATCH --error=/workspace-vast/annas/logs/conversation_v4_l30_opp_ortho_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=general

cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/.secrets/load_secrets.sh
source .venv/bin/activate

echo "V4: Layer 30, Opposite-pair, Orthogonalized"
python3 probes/ua_emotion_disentangle/analyze_conversation_projections_v4.py \
    --layer 30 --method opposite --orthogonalize

echo "Completed: $(date)"
