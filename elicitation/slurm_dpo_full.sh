#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --job-name=dpo_full
#SBATCH --output=/workspace-vast/annas/logs/dpo_full_%j.out
#SBATCH --error=/workspace-vast/annas/logs/dpo_full_%j.err
#SBATCH --time=06:00:00

# Full DPO training with 283 pairs (all 10 prompts)

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools

# Use believe-it-or-not venv for TRL/PEFT
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

echo "=========================================="
echo "Full DPO Training (283 pairs)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=========================================="

python -u elicitation/finetune_dpo_trl.py \
    "google/gemma-3-27b-it" \
    "elicitation/outputs/dpo_pairs_full.jsonl" \
    "/workspace-vast/annas/models/gemma3-27b-dpo-calm-full" \
    --lr 5e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 2048 \
    --max_prompt_length 1024 \
    --lora_r 64 \
    --lora_alpha 64 \
    --beta 0.1

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "DPO training complete!"
else
    echo "DPO training failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
