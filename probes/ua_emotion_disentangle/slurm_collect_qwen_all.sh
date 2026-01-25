#!/bin/bash
#SBATCH --job-name=qwen_all
#SBATCH --output=logs/qwen_all_%j.log
#SBATCH --error=logs/qwen_all_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00

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

# Create logs and data directories
mkdir -p /workspace-vast/annas/git/research-tools/probes/ua_emotion_disentangle/logs
mkdir -p /workspace-vast/annas/git/research-tools/probes/ua_emotion_disentangle/data

# Change to directory
cd /workspace-vast/annas/git/research-tools/probes/ua_emotion_disentangle

echo "=========================================="
echo "PART 1: UA Emotions (Qwen-corrected templates)"
echo "=========================================="
echo

# First, remove the incorrectly collected file
rm -f data/qwen3_32b_ua_emotions.h5

# Collect UA emotions with CORRECTED Qwen templates
python collect_qwen_corrected.py \
    --layers "22-42" \
    --output "data/qwen3_32b_ua_emotions_corrected.h5"

echo
echo "=========================================="
echo "PART 2: Texts Dataset"
echo "=========================================="
echo

# Collect text activations
python collect_texts_qwen.py \
    --input "/workspace-vast/annas/git/research-tools/outputs/data/texts_combined_pairs.jsonl" \
    --layers "22-42" \
    --output "data/qwen3_32b_texts_combined.h5"

echo
echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $?"
echo "=========================================="
