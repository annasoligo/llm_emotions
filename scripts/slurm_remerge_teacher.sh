#!/bin/bash
#SBATCH --partition=highram
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --job-name=remerge_teacher
#SBATCH --output=/workspace-vast/annas/logs/remerge_teacher_%j.out
#SBATCH --error=/workspace-vast/annas/logs/remerge_teacher_%j.err
#SBATCH --time=4:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "=========================================="
echo "Re-merging Teacher Mode model with vision encoder"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

python scripts/remerge_and_upload.py \
    --adapter-path /workspace-vast/annas/models/gemma3-27b-teacher-mode/2026-01-14_16-51-07 \
    --output-name gemma3-27b-teacher-mode-merged \
    --base-model google/gemma-3-27b-it

echo "=========================================="
echo "Done"
echo "=========================================="
