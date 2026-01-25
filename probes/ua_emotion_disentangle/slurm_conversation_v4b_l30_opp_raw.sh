#!/bin/bash
#SBATCH --job-name=v4b_l30_opp_r
#SBATCH --output=/workspace-vast/annas/logs/conversation_v4b_l30_opp_raw_%j.out
#SBATCH --error=/workspace-vast/annas/logs/conversation_v4b_l30_opp_raw_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=general

cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/.secrets/load_secrets.sh
source .venv/bin/activate

echo "V4b: Layer 30, Opposite-pair, Raw (Raw cosine sim)"
python3 probes/ua_emotion_disentangle/analyze_conversation_projections_v4b.py \
    --layer 30 --method opposite

echo "Completed: $(date)"
