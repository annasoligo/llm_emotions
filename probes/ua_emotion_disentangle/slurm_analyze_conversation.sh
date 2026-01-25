#!/bin/bash
#SBATCH --job-name=conv_analysis
#SBATCH --output=/workspace-vast/annas/logs/conversation_analysis_%j.out
#SBATCH --error=/workspace-vast/annas/logs/conversation_analysis_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=general

cd /workspace-vast/annas/git/research-tools

source /workspace-vast/annas/.secrets/load_secrets.sh
source .venv/bin/activate

echo "========================================"
echo "CONVERSATION PROJECTION ANALYSIS"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo ""

python3 probes/ua_emotion_disentangle/analyze_conversation_projections.py

echo ""
echo "Completed: $(date)"
