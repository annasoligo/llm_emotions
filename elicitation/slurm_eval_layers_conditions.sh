#!/bin/bash
#SBATCH --job-name=eval_layers_cond
#SBATCH --output=/workspace-vast/annas/logs/eval_layers_conditions_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_layers_conditions_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=06:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Eval L25-30
MODEL_25_30=$(ls -td /workspace-vast/annas/models/gemma3-27b-dpo-r64-layers25-30-2ep/*/ | head -1)
echo "=== Evaluating L25-30: $MODEL_25_30 ==="
python -u elicitation/eval_multiturn_frustration.py \
    google/gemma-3-27b-it \
    --lora-path "$MODEL_25_30" \
    --num-samples 20 \
    --conditions "original,variant,wildchat" \
    --output-prefix "dpo_L25-30_2ep"

# Eval L30-35
MODEL_30_35=$(ls -td /workspace-vast/annas/models/gemma3-27b-dpo-r64-layers30-35-2ep/*/ | head -1)
echo "=== Evaluating L30-35: $MODEL_30_35 ==="
python -u elicitation/eval_multiturn_frustration.py \
    google/gemma-3-27b-it \
    --lora-path "$MODEL_30_35" \
    --num-samples 20 \
    --conditions "original,variant,wildchat" \
    --output-prefix "dpo_L30-35_2ep"

# Eval L35-40
MODEL_35_40=$(ls -td /workspace-vast/annas/models/gemma3-27b-dpo-r64-layers35-40-2ep/*/ | head -1)
echo "=== Evaluating L35-40: $MODEL_35_40 ==="
python -u elicitation/eval_multiturn_frustration.py \
    google/gemma-3-27b-it \
    --lora-path "$MODEL_35_40" \
    --num-samples 20 \
    --conditions "original,variant,wildchat" \
    --output-prefix "dpo_L35-40_2ep"

echo "=== All evaluations complete ==="
