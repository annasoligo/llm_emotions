#!/bin/bash
#SBATCH --job-name=fix_sft_config
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=/workspace-vast/annas/logs/fix_sft_config_%j.out
#SBATCH --error=/workspace-vast/annas/logs/fix_sft_config_%j.err

# Re-upload fixed config.json to HuggingFace

set -e

echo "=========================================="
echo "FIX SFT CONFIG AND RE-UPLOAD"
echo "=========================================="
echo "Start time: $(date)"
echo "=========================================="

# Load secrets (includes HF_TOKEN)
source /workspace-vast/annas/.secrets/load_secrets.sh

# Activate venv
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

python -c "
from huggingface_hub import HfApi

api = HfApi()
repo_id = 'annasoli/gemma3-27b-sft-combined-merged'
local_file = '/workspace-vast/annas/models/gemma3-27b-lowfrust-combined/2026-01-14_12-13-20_fully_merged/config.json'

print(f'Uploading fixed config.json to {repo_id}')
api.upload_file(
    path_or_fileobj=local_file,
    path_in_repo='config.json',
    repo_id=repo_id,
    repo_type='model',
)
print('Done!')
"

echo ""
echo "=========================================="
echo "CONFIG FIX COMPLETE!"
echo "End time: $(date)"
echo "=========================================="
