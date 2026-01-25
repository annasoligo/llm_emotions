#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --job-name=upload_L3040
#SBATCH --output=/workspace-vast/annas/logs/upload_layers30-40_v2_%j.out
#SBATCH --error=/workspace-vast/annas/logs/upload_layers30-40_v2_%j.err
#SBATCH --time=01:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME="/workspace-vast/annas/.cache/huggingface"

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python3 << 'PYTHON'
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import torch

api = HfApi()

# Create the repo first
print("Creating repo...")
api.create_repo("annasoli/gemma3-27b-dpo-r64-layers30-40-2ep-merged", repo_type="model", exist_ok=True)

# Load and push
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "/workspace-vast/annas/models/gemma3-27b-dpo-r64-layers30-40-2ep-merged",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("/workspace-vast/annas/models/gemma3-27b-dpo-r64-layers30-40-2ep-merged", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("/workspace-vast/annas/models/gemma3-27b-dpo-r64-layers30-40-2ep-merged", trust_remote_code=True)

print("Pushing to hub...")
model.push_to_hub("annasoli/gemma3-27b-dpo-r64-layers30-40-2ep-merged", private=False)
tokenizer.push_to_hub("annasoli/gemma3-27b-dpo-r64-layers30-40-2ep-merged", private=False)
processor.push_to_hub("annasoli/gemma3-27b-dpo-r64-layers30-40-2ep-merged", private=False)
print("Done!")
PYTHON
