#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=dpo_train
#SBATCH --output=/workspace-vast/annas/logs/dpo_train_%j.out
#SBATCH --error=/workspace-vast/annas/logs/dpo_train_%j.err
#SBATCH --time=04:00:00

# DPO finetuning for frustration reduction

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools

if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

# Combine multiple calm data sources
CALM_FILES=""
for f in \
    "elicitation/outputs/diverse_calm_20260115_081258.jsonl" \
    "elicitation/outputs/multiturn_low_frustration_20260114_110500.jsonl" \
    "elicitation/outputs/teacher_mode_20260114_161036.jsonl"; do
    if [ -f "$f" ]; then
        CALM_FILES="$CALM_FILES --calm-file $f"
    fi
done

# Combine multiple frustrated data sources (vanilla eval)
FRUSTRATED_FILES=""
for f in \
    "elicitation/outputs/eval_multiturn/eval_original_gemma-3-27b-it_20260114_102904.jsonl" \
    "elicitation/outputs/eval_multiturn/eval_variant_gemma-3-27b-it_20260114_102904.jsonl"; do
    if [ -f "$f" ]; then
        FRUSTRATED_FILES="$FRUSTRATED_FILES --frustrated-file $f"
    fi
done

echo "=========================================="
echo "Step 1: Preparing DPO data"
echo "=========================================="
echo "Calm files: $CALM_FILES"
echo "Frustrated files: $FRUSTRATED_FILES"

python elicitation/prepare_dpo_data.py \
    $CALM_FILES \
    $FRUSTRATED_FILES \
    --output-file "elicitation/outputs/dpo_pairs.jsonl" \
    --max-calm-rating 1 \
    --min-frustrated-rating 3 \
    --max-pairs-per-key 100

echo ""
echo "=========================================="
echo "Step 2: DPO Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=========================================="

# Switch to believe-it-or-not venv for unsloth
cd /workspace-vast/annas/git/believe-it-or-not
deactivate 2>/dev/null || true
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

python -u /workspace-vast/annas/git/research-tools/elicitation/finetune_dpo_unsloth.py \
    "google/gemma-3-27b-it" \
    "/workspace-vast/annas/git/research-tools/elicitation/outputs/dpo_pairs.jsonl" \
    "/workspace-vast/annas/models/gemma3-27b-dpo-calm" \
    --lr 5e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 2048 \
    --max_prompt_length 1024 \
    --lora_r 64 \
    --lora_alpha 64 \
    --beta 0.1 \
    --warmup_ratio 0.1 \
    --logging_steps 10

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "DPO training complete!"
    echo "Model saved to: /workspace-vast/annas/models/gemma3-27b-dpo-calm"
else
    echo "DPO training failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
