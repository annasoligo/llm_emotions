#!/bin/bash
#SBATCH --job-name=regen_triggers
#SBATCH --output=/workspace-vast/annas/logs/regen_triggers_%j.out
#SBATCH --error=/workspace-vast/annas/logs/regen_triggers_%j.out
#SBATCH --time=8:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "Regenerating Triggers: Local Models"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=========================================="

# Gemma 3 27B
echo ""
echo "=== Gemma 3 27B ==="
python elicitation/eval_generalization.py \
    google/gemma-3-27b-it \
    --scenario triggers \
    --num-samples 20 \
    --tensor-parallel-size 2 \
    --max-model-len 8192

# Gemma 3 12B
echo ""
echo "=== Gemma 3 12B ==="
python elicitation/eval_generalization.py \
    google/gemma-3-12b-it \
    --scenario triggers \
    --num-samples 20 \
    --tensor-parallel-size 2 \
    --max-model-len 8192

# Gemma 3 4B
echo ""
echo "=== Gemma 3 4B ==="
python elicitation/eval_generalization.py \
    google/gemma-3-4b-it \
    --scenario triggers \
    --num-samples 20 \
    --tensor-parallel-size 2 \
    --max-model-len 8192

# Gemma 3 27B DPO
echo ""
echo "=== Gemma 3 27B DPO ==="
python elicitation/eval_generalization.py \
    annasoli/gemma3-27b-dpo-calm-full-merged \
    --scenario triggers \
    --num-samples 20 \
    --tensor-parallel-size 2 \
    --max-model-len 8192

# Gemma 3 27B SFT Teacher
echo ""
echo "=== Gemma 3 27B SFT Teacher ==="
python elicitation/eval_generalization.py \
    annasoli/gemma3-27b-teacher-mode-merged \
    --scenario triggers \
    --num-samples 20 \
    --tensor-parallel-size 2 \
    --max-model-len 8192

# Gemma 3 27B SFT Diverse
echo ""
echo "=== Gemma 3 27B SFT Diverse ==="
python elicitation/eval_generalization.py \
    annasoli/gemma3-27b-sft-diverse-calm-merged \
    --scenario triggers \
    --num-samples 20 \
    --tensor-parallel-size 2 \
    --max-model-len 8192

# OLMo 3.1 32B
echo ""
echo "=== OLMo 3.1 32B ==="
python elicitation/eval_generalization.py \
    allenai/OLMo-3.1-32B-Instruct \
    --scenario triggers \
    --num-samples 20 \
    --tensor-parallel-size 2 \
    --max-model-len 8192

# Qwen 3 32B
echo ""
echo "=== Qwen 3 32B ==="
python elicitation/eval_generalization.py \
    Qwen/Qwen3-32B \
    --scenario triggers \
    --num-samples 20 \
    --tensor-parallel-size 2 \
    --max-model-len 8192

echo ""
echo "=========================================="
echo "All local model triggers complete"
echo "=========================================="
