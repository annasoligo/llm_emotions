#!/bin/bash
#SBATCH --job-name=conv_l40
#SBATCH --output=/workspace-vast/annas/logs/conversation_layer40_%j.out
#SBATCH --error=/workspace-vast/annas/logs/conversation_layer40_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00

echo "========================================"
echo "CONVERSATION PROJECTION ANALYSIS - LAYER 40"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

cd /workspace-vast/annas/git/research-tools

python3 probes/ua_emotion_disentangle/analyze_conversation_multilayer.py --layer 40

echo
echo "Completed: $(date)"
