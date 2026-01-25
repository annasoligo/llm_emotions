#!/bin/bash
#SBATCH --job-name=ua_olmo
#SBATCH --output=logs/ua_olmo_%j.log
#SBATCH --error=logs/ua_olmo_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=6:00:00

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo

# Load secrets
echo "Loading secrets..."
source /workspace-vast/annas/.secrets/load_secrets.sh
echo "✓ Secrets loaded"
echo

# Activate virtual environment
echo "Activating virtual environment..."
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate
echo "✓ Virtual environment activated"
echo

# Create logs directory if needed
mkdir -p /workspace-vast/annas/git/research-tools/probes/ua_emotion_disentangle/logs

# Change to directory
cd /workspace-vast/annas/git/research-tools/probes/ua_emotion_disentangle

# Run collection for OLMo-3-32B-Think
# OLMo-3-32B has 64 layers, using middle 21 layers (22-42)
python collect_olmo_corrected.py \
    --model "allenai/Olmo-3-32B-Think" \
    --layers "22-42" \
    --output "data/olmo_32b_think_ua_emotions.h5" \
    --dtype bfloat16

echo
echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $?"
echo "=========================================="
