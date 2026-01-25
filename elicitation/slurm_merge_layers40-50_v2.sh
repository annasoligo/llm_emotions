#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --job-name=merge_L4050
#SBATCH --output=/workspace-vast/annas/logs/merge_layers40-50_v2_%j.out
#SBATCH --error=/workspace-vast/annas/logs/merge_layers40-50_v2_%j.err
#SBATCH --time=01:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME="/workspace-vast/annas/.cache/huggingface"

cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python3 << 'PYTHON'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from peft import PeftModel

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-27b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it", trust_remote_code=True)

print("Loading processor...")
processor = AutoProcessor.from_pretrained("google/gemma-3-27b-it", trust_remote_code=True)

print("Loading adapter...")
model = PeftModel.from_pretrained(model, "/workspace-vast/annas/models/gemma3-27b-dpo-r64-layers40-50-2ep/2026-01-18_11-29-53")

print("Merging...")
model = model.merge_and_unload()

print("Saving locally...")
model.save_pretrained("/workspace-vast/annas/models/gemma3-27b-dpo-r64-layers40-50-2ep-merged")
tokenizer.save_pretrained("/workspace-vast/annas/models/gemma3-27b-dpo-r64-layers40-50-2ep-merged")
processor.save_pretrained("/workspace-vast/annas/models/gemma3-27b-dpo-r64-layers40-50-2ep-merged")

print("Pushing to hub...")
model.push_to_hub("annasoli/gemma3-27b-dpo-r64-layers40-50-2ep-merged", private=False)
tokenizer.push_to_hub("annasoli/gemma3-27b-dpo-r64-layers40-50-2ep-merged", private=False)
processor.push_to_hub("annasoli/gemma3-27b-dpo-r64-layers40-50-2ep-merged", private=False)

print("Done!")
PYTHON
