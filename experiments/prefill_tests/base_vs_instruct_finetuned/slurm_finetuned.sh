#!/bin/bash
#SBATCH --job-name=ft_frust
#SBATCH --output=/workspace-vast/annas/git/research-tools/experiments/base_vs_instruct_finetuned/logs/%j.log
#SBATCH --error=/workspace-vast/annas/git/research-tools/experiments/base_vs_instruct_finetuned/logs/%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=140G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Finetuned model frustration experiment
# Compares: gemma instruct vs base vs finetuned (annasoli/gemma3-27b-dpo-calm-full)

echo "========================================"
echo "FINETUNED MODEL FRUSTRATION EXPERIMENT"
echo "Models: instruct, base, finetuned (annasoli/gemma3-27b-dpo-calm-full)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================"

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Activated venv"
fi

export HF_HOME=/workspace-vast/pretrained_ckpts

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "========================================"

# Run ONLY finetuned model with BOTH truncation types
# (we already have instruct/base results from earlier experiments)
python experiments/base_vs_instruct_finetuned.py \
    --model-type finetuned \
    --truncation both \
    --phase all

if [ $? -ne 0 ]; then
    echo "ERROR: Experiment failed"
    exit 1
fi

echo "✓ Finetuned experiment complete"
