#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=neutral_convs
#SBATCH --output=/workspace-vast/annas/logs/%j.out
#SBATCH --error=/workspace-vast/annas/logs/%j.err

# Generate neutral conversation paraphrases using Batch API

# Load authentication
source /workspace-vast/annas/.secrets/load_secrets.sh
export HF_HOME=/workspace-vast/pretrained_ckpts

# Activate environment
cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Run batch generation
python -m probes.scripts.generate_neutral_conversations_batch \
    --input data/conversations2.jsonl \
    --output data/conversations2_neutral.jsonl \
    --batch_requests batches/neutral_conv_requests.jsonl \
    --batch_results batches/neutral_conv_results.jsonl \
    --poll_interval 120

echo ""
echo "Neutral conversations generated!"
echo "Output: data/conversations2_neutral.jsonl"
