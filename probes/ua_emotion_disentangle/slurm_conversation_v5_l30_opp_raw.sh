#!/bin/bash
#SBATCH --job-name=v5_l30_opp_r
#SBATCH --output=/workspace-vast/annas/logs/conversation_v5_l30_opp_raw_%j.out
#SBATCH --error=/workspace-vast/annas/logs/conversation_v5_l30_opp_raw_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=general

cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/.secrets/load_secrets.sh
source .venv/bin/activate

echo "V5: Layer 30, Opposite-pair, Raw (Global mean + Neutral proj)"
python3 probes/ua_emotion_disentangle/analyze_conversation_projections_v5.py \
    --layer 30 --method opposite

echo "Completed: $(date)"
