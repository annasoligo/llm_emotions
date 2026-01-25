#!/bin/bash
#SBATCH --job-name=ua_qwen3
#SBATCH --output=logs/ua_qwen3_%j.log
#SBATCH --error=logs/ua_qwen3_%j.log
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

# Run collection for Qwen3-32B using hooks (more robust than nnsight)
# Layers 22-42 (middle 21 layers of 64 total)
python collect_multimodel_hooks.py \
    --model "Qwen/Qwen3-32B" \
    --layers "22-42" \
    --output "data/qwen3_32b_ua_emotions.h5" \
    --dtype bfloat16

echo
echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $?"
echo "=========================================="
