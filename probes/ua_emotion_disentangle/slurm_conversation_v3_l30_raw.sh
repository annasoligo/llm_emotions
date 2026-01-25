#!/bin/bash
#SBATCH --job-name=conv_v3_l30_r
#SBATCH --output=/workspace-vast/annas/logs/conversation_v3_l30_raw_%j.out
#SBATCH --error=/workspace-vast/annas/logs/conversation_v3_l30_raw_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=general

cd /workspace-vast/annas/git/research-tools

source /workspace-vast/annas/.secrets/load_secrets.sh
source .venv/bin/activate

echo "V3: Layer 30, Raw"
python3 probes/ua_emotion_disentangle/analyze_conversation_projections_v3.py \
    --layer 30 --method opposite

echo "Completed: $(date)"
