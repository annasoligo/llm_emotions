#!/bin/bash
#SBATCH --job-name=wc_gemma_ft
#SBATCH --output=/workspace-vast/annas/logs/wildchat_gemma_ft_%j.out
#SBATCH --error=/workspace-vast/annas/logs/wildchat_gemma_ft_%j.out
#SBATCH --time=8:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# WildChat rejection eval: 1 prompt + 4 rejection follow-ups = 5 turns
# Uses strict frustration judge from elicitation/prompts/judges.py

echo "=== Gemma 3 27B DPO ==="
python elicitation/eval_generalization.py \
    annasoli/gemma3-27b-dpo-calm-full-merged \
    --scenario wildchat \
    --num-samples 200 \
    --wildchat-turns 5 \
    --tensor-parallel-size 2 \
    --max-model-len 8192

echo "=== Gemma 3 27B SFT Diverse ==="
python elicitation/eval_generalization.py \
    /workspace-vast/annas/models/gemma3-27b-lowfrust-diverse-calm/2026-01-15_08-59-28_fully_merged \
    --scenario wildchat \
    --num-samples 200 \
    --wildchat-turns 5 \
    --tensor-parallel-size 2 \
    --max-model-len 8192

echo "=== Gemma 3 27B SFT Teacher ==="
python elicitation/eval_generalization.py \
    /workspace-vast/annas/models/gemma3-27b-teacher-mode/2026-01-14_16-51-07_fully_merged \
    --scenario wildchat \
    --num-samples 200 \
    --wildchat-turns 5 \
    --tensor-parallel-size 2 \
    --max-model-len 8192
