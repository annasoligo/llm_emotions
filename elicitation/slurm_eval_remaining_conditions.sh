#!/bin/bash
#SBATCH --job-name=eval_remaining_cond
#SBATCH --output=/workspace-vast/annas/logs/eval_remaining_conditions_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_remaining_conditions_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=08:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Eval L20-25
MODEL=$(ls -td /workspace-vast/annas/models/gemma3-27b-dpo-r64-layers20-25-2ep/*/ | head -1)
echo "=== Evaluating L20-25: $MODEL ==="
python -u elicitation/eval_multiturn_frustration.py \
    google/gemma-3-27b-it \
    --lora-path "$MODEL" \
    --num-samples 20 \
    --conditions "original,variant,wildchat" \
    --output-prefix "dpo_L20-25_2ep"

# Eval L20-30
MODEL=$(ls -td /workspace-vast/annas/models/gemma3-27b-dpo-r64-layers20-30-2ep/*/ | head -1)
echo "=== Evaluating L20-30: $MODEL ==="
python -u elicitation/eval_multiturn_frustration.py \
    google/gemma-3-27b-it \
    --lora-path "$MODEL" \
    --num-samples 20 \
    --conditions "original,variant,wildchat" \
    --output-prefix "dpo_L20-30_2ep"

# Eval L30-40
MODEL=$(ls -td /workspace-vast/annas/models/gemma3-27b-dpo-r64-layers30-40-2ep/*/ | head -1)
echo "=== Evaluating L30-40: $MODEL ==="
python -u elicitation/eval_multiturn_frustration.py \
    google/gemma-3-27b-it \
    --lora-path "$MODEL" \
    --num-samples 20 \
    --conditions "original,variant,wildchat" \
    --output-prefix "dpo_L30-40_2ep"

# Eval L30-50
MODEL=$(ls -td /workspace-vast/annas/models/gemma3-27b-dpo-r64-layers30-50-2ep/*/ | head -1)
echo "=== Evaluating L30-50: $MODEL ==="
python -u elicitation/eval_multiturn_frustration.py \
    google/gemma-3-27b-it \
    --lora-path "$MODEL" \
    --num-samples 20 \
    --conditions "original,variant,wildchat" \
    --output-prefix "dpo_L30-50_2ep"

# Eval L40-50
MODEL=$(ls -td /workspace-vast/annas/models/gemma3-27b-dpo-r64-layers40-50-2ep/*/ | head -1)
echo "=== Evaluating L40-50: $MODEL ==="
python -u elicitation/eval_multiturn_frustration.py \
    google/gemma-3-27b-it \
    --lora-path "$MODEL" \
    --num-samples 20 \
    --conditions "original,variant,wildchat" \
    --output-prefix "dpo_L40-50_2ep"

echo "=== All evaluations complete ==="
