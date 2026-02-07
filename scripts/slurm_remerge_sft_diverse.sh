#!/bin/bash
#SBATCH --partition=highram
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --job-name=remerge_sft
#SBATCH --output=/workspace-vast/annas/logs/remerge_sft_%j.out
#SBATCH --error=/workspace-vast/annas/logs/remerge_sft_%j.err
#SBATCH --time=4:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "=========================================="
echo "Re-merging SFT Diverse Calm model with vision encoder"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

python scripts/remerge_and_upload.py \
    --adapter-path /workspace-vast/annas/models/gemma3-27b-lowfrust-diverse-calm/2026-01-15_08-59-28 \
    --output-name gemma3-27b-sft-diverse-calm-merged \
    --base-model google/gemma-3-27b-it

echo "=========================================="
echo "Done"
echo "=========================================="
