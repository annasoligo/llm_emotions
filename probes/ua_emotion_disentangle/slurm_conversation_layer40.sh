#!/bin/bash
#SBATCH --job-name=conv_l40
#SBATCH --output=/workspace-vast/annas/logs/conversation_layer40_%j.out
#SBATCH --error=/workspace-vast/annas/logs/conversation_layer40_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/git/research-tools/.venv/bin/activate

echo "========================================"
echo "CONVERSATION PROJECTION ANALYSIS - LAYER 40"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"

python3 probes/ua_emotion_disentangle/analyze_conversation_layer40.py

echo ""
echo "Completed: $(date)"
