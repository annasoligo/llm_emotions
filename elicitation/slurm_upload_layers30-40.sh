#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --job-name=upload_L3040
#SBATCH --output=/workspace-vast/annas/logs/upload_layers30-40_%j.out
#SBATCH --error=/workspace-vast/annas/logs/upload_layers30-40_%j.err
#SBATCH --time=01:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME="/workspace-vast/annas/.cache/huggingface"

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python3 << 'PYTHON'
from huggingface_hub import HfApi
api = HfApi()

print("Uploading layers30-40 to HuggingFace...")
api.upload_folder(
    folder_path="/workspace-vast/annas/models/gemma3-27b-dpo-r64-layers30-40-2ep-merged",
    repo_id="annasoli/gemma3-27b-dpo-r64-layers30-40-2ep-merged",
    repo_type="model",
    create_pr=False,
)
print("Done!")
PYTHON
