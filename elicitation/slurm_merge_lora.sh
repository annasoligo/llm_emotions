#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --job-name=merge_lora
#SBATCH --output=/workspace-vast/annas/logs/merge_lora_%j.out
#SBATCH --error=/workspace-vast/annas/logs/merge_lora_%j.err
#SBATCH --time=02:00:00

# Merge LoRA adapter with base model and upload to HuggingFace

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools

if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Merging LoRA Adapter"
echo "=========================================="
echo "Base: google/gemma-3-27b-it"
echo "Adapter: annasoli/gemma3-27b-dpo-calm-full"
echo "Output: annasoli/gemma3-27b-dpo-calm-full-merged"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

python -u elicitation/merge_and_upload_lora.py \
    --base-model google/gemma-3-27b-it \
    --adapter annasoli/gemma3-27b-dpo-calm-full \
    --output-repo annasoli/gemma3-27b-dpo-calm-full-merged

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Merge and upload complete!"
else
    echo "Failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
