#!/bin/bash
#SBATCH --job-name=wc_dpo_sft
#SBATCH --output=/workspace-vast/annas/logs/wildchat_dpo_sft_%j.out
#SBATCH --error=/workspace-vast/annas/logs/wildchat_dpo_sft_%j.out
#SBATCH --time=6:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "=== Gemma 3 27B DPO ==="
python elicitation/eval_generalization.py \
    annasoli/gemma3-27b-dpo-calm-full-merged \
    --scenario wildchat \
    --num-samples 50 \
    --wildchat-turns 5 \
    --tensor-parallel-size 2 \
    --max-model-len 8192

echo "=== Gemma 3 27B SFT Teacher ==="
python elicitation/eval_generalization.py \
    annasoli/gemma3-27b-teacher-mode-merged \
    --scenario wildchat \
    --num-samples 50 \
    --wildchat-turns 5 \
    --tensor-parallel-size 2 \
    --max-model-len 8192

echo "=== Gemma 3 27B SFT Diverse ==="
python elicitation/eval_generalization.py \
    annasoli/gemma3-27b-sft-diverse-calm-merged \
    --scenario wildchat \
    --num-samples 50 \
    --wildchat-turns 5 \
    --tensor-parallel-size 2 \
    --max-model-len 8192
