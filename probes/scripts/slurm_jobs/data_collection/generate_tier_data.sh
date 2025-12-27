#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=data_augmentation
#SBATCH --output=/workspace-vast/annas/git/research-tools/probes/logs/%j.out
#SBATCH --error=/workspace-vast/annas/git/research-tools/probes/logs/%j.err

# CPU-only job example - no GPU needed
# Note: No --gres flag = no GPU requested

# Load authentication
source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME=/workspace-vast/pretrained_ckpts

# Activate environment if needed
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

python probes/scripts/generate_data.py \
  --mode pairs \
  --output data/pairs2.jsonl \
  --n_per_combo 200 \
  --claude_model claude-3-5-haiku-20241022 \
  --claude_max_concurrent 10 \
  --max_retries 10 \
  --use_batch_api

    

