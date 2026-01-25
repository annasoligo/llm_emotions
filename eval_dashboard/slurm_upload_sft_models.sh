#!/bin/bash
#SBATCH --job-name=upload_sft
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/annas/logs/upload_sft_%j.out
#SBATCH --error=/workspace-vast/annas/logs/upload_sft_%j.err

# Upload merged SFT models to HuggingFace Hub

set -e

echo "=========================================="
echo "UPLOAD SFT MODELS TO HUGGINGFACE"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

# Load secrets (includes HF_TOKEN)
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate venv
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

cd /workspace-vast/annas/git/research-tools

# Model 1: SFT Combined (all layers, 1 epoch)
echo ""
echo "Uploading Model 1: sft_combined"
echo "================================"
python -c "
from huggingface_hub import HfApi
import os

api = HfApi()
repo_id = 'annasoli/gemma3-27b-sft-combined-merged'
local_dir = '/workspace-vast/annas/models/gemma3-27b-lowfrust-combined/2026-01-14_12-13-20_fully_merged'

print(f'Creating repo: {repo_id}')
api.create_repo(repo_id, exist_ok=True, private=False)

print(f'Uploading from: {local_dir}')
api.upload_folder(
    folder_path=local_dir,
    repo_id=repo_id,
    repo_type='model',
)
print(f'Done uploading {repo_id}')
"

# Model 2: SFT Last20 3ep
echo ""
echo "Uploading Model 2: sft_last20_3ep"
echo "=================================="
python -c "
from huggingface_hub import HfApi
import os

api = HfApi()
repo_id = 'annasoli/gemma3-27b-sft-last20-3ep-merged'
local_dir = '/workspace-vast/annas/models/gemma3-27b-sft-r64-last20-3ep-merged'

print(f'Creating repo: {repo_id}')
api.create_repo(repo_id, exist_ok=True, private=False)

print(f'Uploading from: {local_dir}')
api.upload_folder(
    folder_path=local_dir,
    repo_id=repo_id,
    repo_type='model',
)
print(f'Done uploading {repo_id}')
"

echo ""
echo "=========================================="
echo "ALL UPLOADS COMPLETE!"
echo "End time: $(date)"
echo "=========================================="
