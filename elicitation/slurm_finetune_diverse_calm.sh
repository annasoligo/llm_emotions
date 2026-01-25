#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=sft_diverse
#SBATCH --output=/workspace-vast/annas/logs/sft_diverse_calm_%j.out
#SBATCH --error=/workspace-vast/annas/logs/sft_diverse_calm_%j.err
#SBATCH --time=04:00:00

# SFT finetuning with diverse_calm + boundary-setting multiturn data

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools

if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Step 1: Preparing diverse calm SFT data"
echo "=========================================="

# Combine diverse_calm + boundary-setting multiturn data
python elicitation/prepare_diverse_calm_sft_data.py \
    --multiturn-file "elicitation/outputs/diverse_calm_20260115_081258.jsonl" \
    --multiturn-file "elicitation/outputs/multiturn_low_frustration_20260114_110500.jsonl" \
    --instruct-file "/workspace-vast/annas/git/believe-it-or-not/data/instruct_data/neutral_v2_997_dolci_500.jsonl" \
    --output-file "elicitation/outputs/diverse_calm_sft_data.jsonl" \
    --max-rating 1 \
    --num-instruct 500

echo ""
echo "=========================================="
echo "Step 2: SFT Finetuning"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=========================================="

# Switch to believe-it-or-not venv for finetuning
cd /workspace-vast/annas/git/believe-it-or-not
deactivate 2>/dev/null || true
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

python -u science_synth_facts/finetuning/finetune_gpu_unsloth.py \
    "google/gemma-3-27b-it" \
    "/workspace-vast/annas/git/research-tools/elicitation/outputs/diverse_calm_sft_data.jsonl" \
    "/workspace-vast/annas/models/gemma3-27b-lowfrust-diverse-calm" \
    --lr 0.0001 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_length 4096 \
    --lora_r 64 \
    --lora_alpha 128 \
    --warmup_steps 50 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 2 \
    --eval_strategy "no" \
    --load_in_4bit True

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "SFT finetuning complete!"
    echo "Model saved to: /workspace-vast/annas/models/gemma3-27b-lowfrust-diverse-calm"
else
    echo "SFT finetuning failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
