#!/bin/bash
#SBATCH --job-name=eval_a256_var
#SBATCH --output=/workspace-vast/annas/logs/eval_last20_a256_variant_wc_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_last20_a256_variant_wc_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=02:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

MODEL=$(ls -td /workspace-vast/annas/models/gemma3-27b-dpo-r64-last20-alpha256-2ep/*/ | head -1)
echo "Model: $MODEL"

python -u elicitation/eval_multiturn_frustration.py \
    google/gemma-3-27b-it \
    --lora-path "$MODEL" \
    --num-samples 20 --num-turns 3 \
    --max-tokens 10000 \
    --conditions variant,wildchat \
    --output-prefix "last20_a256"
