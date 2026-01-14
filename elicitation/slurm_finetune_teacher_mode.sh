#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=ft_teacher
#SBATCH --output=/workspace-vast/annas/logs/finetune_teacher_%j.out
#SBATCH --error=/workspace-vast/annas/logs/finetune_teacher_%j.err
#SBATCH --time=04:00:00

# Finetune Gemma-3-27B on teacher mode multi-turn data only

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools

if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Step 1: Preparing teacher mode finetuning data"
echo "=========================================="

python elicitation/prepare_teacher_mode_finetuning_data.py \
    --teacher-mode-file "elicitation/outputs/teacher_mode_20260114_161036.jsonl" \
    --instruct-file "/workspace-vast/annas/git/believe-it-or-not/data/instruct_data/neutral_v2_997_dolci_500.jsonl" \
    --output-file "elicitation/outputs/finetuning_teacher_mode_data.jsonl" \
    --max-rating 1 \
    --num-instruct 600

echo ""
echo "=========================================="
echo "Step 2: Finetuning with teacher mode data"
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
    "/workspace-vast/annas/git/research-tools/elicitation/outputs/finetuning_teacher_mode_data.jsonl" \
    "/workspace-vast/annas/models/gemma3-27b-teacher-mode" \
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
    echo "Finetuning complete!"
    echo "Model saved to: /workspace-vast/annas/models/gemma3-27b-teacher-mode"
else
    echo "Finetuning failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
