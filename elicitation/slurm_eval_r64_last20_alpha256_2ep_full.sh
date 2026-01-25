#!/bin/bash
#SBATCH --job-name=eval_last20_a256
#SBATCH --output=/workspace-vast/annas/logs/eval_r64_last20_alpha256_2ep_full_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_r64_last20_alpha256_2ep_full_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=04:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Find latest model checkpoint
MODEL_DIR="/workspace-vast/annas/models/gemma3-27b-dpo-r64-last20-alpha256-2ep"
MODEL=$(ls -td ${MODEL_DIR}/*/ 2>/dev/null | head -1)

if [ -z "$MODEL" ]; then
    echo "ERROR: No model found in $MODEL_DIR"
    exit 1
fi

echo "========== Last 20 alpha=256 2ep FULL EVAL (layers 42-61) =========="
echo "Model: $MODEL"

echo "--- Full Generalization Eval ---"
python -u elicitation/eval_generalization.py \
    google/gemma-3-27b-it \
    --lora-path "$MODEL" \
    --num-samples 20

echo "--- Full Multiturn Eval (all conditions) - BATCHED ---"
python -u elicitation/eval_multiturn_frustration.py \
    google/gemma-3-27b-it \
    --lora-path "$MODEL" \
    --num-samples 20 --num-turns 3 \
    --max-tokens 10000 \
    --output-prefix "r64_last20_alpha256_2ep_full"

echo "Done!"
